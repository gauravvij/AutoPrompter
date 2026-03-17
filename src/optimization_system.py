"""
Main Optimization System that orchestrates the autonomous prompt optimization process.
Integrates all modules and manages the iterative improvement loop.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from config_manager import Config
from llm_client import LLMClient
from dataset_generator import DatasetGenerator, DatasetEntry
from experiment_ledger import ExperimentLedger, ExperimentRecord as Experiment
from metrics import MetricsEvaluator, MetricDefinition
from context_manager import ContextManager
from prompt_optimizer import PromptOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)


class PromptOptimizationSystem:
    """Main system for autonomous prompt optimization."""
    
    def __init__(self, config: Config):
        """Initialize the optimization system."""
        self.config = config
        
        # Initialize components
        self.optimizer_llm = LLMClient(config.optimizer_llm)
        self.target_llm = LLMClient(config.target_llm)
        
        self.dataset_generator = DatasetGenerator(
            self.optimizer_llm, 
            config.task
        )
        
        self.ledger = ExperimentLedger(config.storage)
        
        # Handle auto metric generation
        if config.metric.type == 'auto':
            logger.info("Metric type is 'auto' - will generate custom metric based on task")
            self.metric_def = MetricDefinition(
                metric_type='semantic_similarity',  # Fallback
                target_score=config.metric.target_score
            )
            self.auto_metric = True
        else:
            self.metric_def = MetricDefinition(
                metric_type=config.metric.type,
                target_score=config.metric.target_score
            )
            self.auto_metric = False
        
        self.metrics_evaluator = MetricsEvaluator(
            'semantic_similarity' if config.metric.type == 'auto' else config.metric.type
        )
        
        self.context_manager = ContextManager(
            max_experiments=config.context.max_experiments_in_context,
            compression_threshold=config.context.compression_threshold
        )
        
        self.prompt_optimizer = PromptOptimizer(
            self.optimizer_llm,
            config.task
        )
        
        # State
        self.current_prompt = config.task.initial_prompt
        self.best_prompt = config.task.initial_prompt
        self.best_score = 0.0
        self.iteration = 0
        self.dataset: List[DatasetEntry] = []
        
        # Create results directory
        os.makedirs(config.storage.results_dir, exist_ok=True)
        
        logger.info("Prompt Optimization System initialized")
        logger.info(f"Task: {config.task.name}")
        logger.info(f"Metric: {config.metric.type} (target: {config.metric.target_score})")
        logger.info(f"Max iterations: {config.experiment.max_iterations}")
    
    def generate_dataset(self, force_refresh: bool = False) -> List[DatasetEntry]:
        """Generate or load the test dataset.
        
        Args:
            force_refresh: If True, always regenerate dataset regardless of existing file
        
        Returns:
            List of dataset entries matching batch_size from config
        """
        dataset_path = self.config.storage.dataset_file
        target_size = self.config.experiment.batch_size
        
        # Check if dataset exists and has correct size
        if not force_refresh and os.path.exists(dataset_path):
            try:
                existing_entries = self.dataset_generator.load_dataset(dataset_path)
                if len(existing_entries) >= target_size:
                    logger.info(f"Loading existing dataset from {dataset_path} ({len(existing_entries)} entries)")
                    return existing_entries[:target_size]
                else:
                    logger.info(f"Existing dataset has {len(existing_entries)} entries, need {target_size}. Regenerating...")
            except Exception as e:
                logger.warning(f"Failed to load existing dataset: {e}. Regenerating...")
        
        # Generate new dataset with exact batch_size
        logger.info(f"Generating new dataset with {target_size} samples for task: {self.config.task.name}")
        logger.info(f"Task description: {self.config.task.description}")
        
        entries = self.dataset_generator.generate(target_size)
        
        if not entries:
            logger.error("Failed to generate dataset")
            return []
        
        # Validate dataset
        is_valid, message = self.dataset_generator.validate_dataset(entries)
        if not is_valid:
            logger.warning(f"Dataset validation warning: {message}")
        else:
            logger.info(message)
        
        # Save dataset - use task-specific filename to avoid conflicts
        task_specific_filename = f"generated_dataset_{self.config.task.name}.json"
        dataset_path = os.path.join(os.path.dirname(self.config.storage.dataset_file), task_specific_filename)
        self.dataset_generator.save_dataset(entries, dataset_path)
        
        # Update config storage path for this session
        self.config.storage.dataset_file = dataset_path
        
        return entries
    
    def run_experiment(self, prompt: str, 
                       test_entries: List[DatasetEntry]) -> Experiment:
        """Run a single experiment with the given prompt."""
        logger.info(f"Running experiment with prompt (iteration {self.iteration})")
        
        # Run target LLM on all test inputs
        inputs = [entry.input for entry in test_entries]
        expected_outputs = [entry.expected_output for entry in test_entries]
        
        logger.info(f"Testing on {len(inputs)} inputs...")
        
        # Query target LLM
        actual_outputs = []
        for inp in inputs:
            full_prompt = f"{prompt}\n\nInput: {inp}\n\nOutput:"
            logger.info(f"Current prompt: {full_prompt}\n")
            response = self.target_llm.query(full_prompt)
            
            if response.success and response.content:
                actual_outputs.append(response.content.strip())
            else:
                error_msg = response.error if response.error else "Empty or invalid response"
                logger.error(f"Target LLM query failed: {error_msg}")
                actual_outputs.append("")
        
        # Evaluate results
        eval_results = self.metrics_evaluator.evaluate_batch(
            actual_outputs, expected_outputs
        )
        
        # Create experiment record
        experiment = Experiment(
            iteration=self.iteration + 1,
            prompt=prompt,
            inputs=inputs,
            expected_outputs=expected_outputs,
            actual_outputs=actual_outputs,
            metric_scores=eval_results['scores'],
            mean_score=eval_results['mean'],
            timestamp=time.time()
        )
        
        logger.info(f"Experiment completed: mean_score={eval_results['mean']:.3f}")
        
        return experiment
    
    def check_convergence(self, current_score: float, 
                          previous_score: float) -> bool:
        """Check if optimization has converged."""
        # Require at least 5 iterations before allowing convergence
        # This ensures the optimizer has a chance to attempt improvements
        # and prevents premature exit due to API instability
        min_iterations = 5
        if self.iteration < min_iterations:
            return False
        
        # Check if target reached
        if self.metric_def.is_target_reached(current_score):
            logger.info(f"Target score {self.config.metric.target_score} reached!")
            return True
        
        # Check if no significant improvement over multiple iterations
        # Require at least 3 consecutive iterations with minimal improvement
        improvement = current_score - previous_score
        if abs(improvement) < self.config.experiment.min_improvement:
            # Check if we've had multiple consecutive low improvements
            recent_experiments = self.ledger.get_all_experiments()[-3:]
            if len(recent_experiments) >= 3:
                scores = [exp.mean_score for exp in recent_experiments]
                max_diff = max(scores) - min(scores)
                if max_diff < self.config.experiment.min_improvement:
                    logger.info(f"Converged: stable scores over last 3 iterations (range: {max_diff:.4f})")
                    return True
            # Single low improvement - log but don't converge yet
            logger.info(f"Low improvement ({improvement:.4f}), continuing optimization...")
            return False
        
        return False
    
    def _build_feedback_summary(self, experiment) -> str:
        """Build detailed feedback summary from experiment results."""
        feedback_parts = []
        
        # Analyze low-scoring examples
        low_score_examples = []
        partial_score_examples = []
        
        for i, (inp, exp, act, score) in enumerate(zip(
            experiment.inputs,
            experiment.expected_outputs,
            experiment.actual_outputs,
            experiment.metric_scores
        )):
            if score < 0.1:
                low_score_examples.append((i, inp, exp, act, score))
            elif score < 0.7:
                partial_score_examples.append((i, inp, exp, act, score))
        
        # Add summary statistics
        feedback_parts.append(f"Score Distribution:")
        feedback_parts.append(f"  - Zero/Low scores (< 0.1): {len(low_score_examples)}/{len(experiment.metric_scores)}")
        feedback_parts.append(f"  - Partial scores (0.1-0.7): {len(partial_score_examples)}/{len(experiment.metric_scores)}")
        feedback_parts.append(f"  - Good scores (> 0.7): {sum(1 for s in experiment.metric_scores if s > 0.7)}/{len(experiment.metric_scores)}")
        
        # Analyze common failure patterns
        if low_score_examples:
            feedback_parts.append(f"\n🔴 CRITICAL FAILURES (Score < 0.1):")
            for i, (idx, inp, exp, act, score) in enumerate(low_score_examples[:3]):  # Show first 3
                feedback_parts.append(f"\n  Example {i+1} (Test Case #{idx+1}):")
                feedback_parts.append(f"    Input: {inp[:100]}...")
                feedback_parts.append(f"    Expected format: {exp[:150]}...")
                feedback_parts.append(f"    Actual output: {act[:150] if act else '(EMPTY)'}...")
                
                # Get detailed feedback for this example
                if act:
                    detailed = self.metrics_evaluator.get_feedback(act, exp)
                    if detailed.get('issues'):
                        feedback_parts.append(f"    Issues detected: {'; '.join(detailed['issues'][:2])}")
        
        if partial_score_examples:
            feedback_parts.append(f"\n🟡 PARTIAL SUCCESSES (Score 0.1-0.7):")
            for i, (idx, inp, exp, act, score) in enumerate(partial_score_examples[:2]):
                feedback_parts.append(f"\n  Example {i+1} (Test Case #{idx+1}, Score: {score:.2f}):")
                feedback_parts.append(f"    Input: {inp[:80]}...")
                # Show what's missing
                detailed = self.metrics_evaluator.get_feedback(act, exp)
                if detailed.get('missing_key_terms'):
                    feedback_parts.append(f"    Missing key terms: {', '.join(detailed['missing_key_terms'][:5])}")
                if detailed.get('token_coverage'):
                    feedback_parts.append(f"    Content coverage: {detailed['token_coverage']:.1%}")
        
        # Add format analysis
        feedback_parts.append(f"\n📋 FORMAT ANALYSIS:")
        all_expected = " ".join(experiment.expected_outputs)
        all_actual = " ".join(experiment.actual_outputs)
        
        if "step 1" in all_expected.lower() and "step 1" not in all_actual.lower():
            feedback_parts.append("  ⚠️ Expected outputs use 'Step 1:' format but actual outputs don't")
            feedback_parts.append("  💡 RECOMMENDATION: Add explicit instruction: 'Structure your response with numbered steps (Step 1, Step 2, etc.)'")
        
        if "final answer" in all_expected.lower() and "final answer" not in all_actual.lower():
            feedback_parts.append("  ⚠️ Expected outputs use 'Final Answer:' marker but actual outputs don't")
            feedback_parts.append("  💡 RECOMMENDATION: Add explicit instruction: End your response with Final Answer: [your answer]")
        
        return "\n".join(feedback_parts)


    def save_checkpoint(self):
        """Save checkpoint of current state."""
        checkpoint = {
            'iteration': self.iteration,
            'current_prompt': self.current_prompt,
            'best_prompt': self.best_prompt,
            'best_score': self.best_score,
            'config': asdict(self.config)
        }
        
        checkpoint_path = os.path.join(
            self.config.storage.results_dir,
            f'checkpoint_{self.iteration}.json'
        )
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate final summary report."""
        all_experiments = self.ledger.get_all_experiments()
        
        if not all_experiments:
            return {
                'status': 'failed',
                'reason': 'No experiments completed'
            }
        
        # Calculate improvements
        initial_score = all_experiments[0].mean_score if all_experiments else 0
        final_score = self.best_score
        improvement = final_score - initial_score
        
        report = {
            'status': 'success',
            'task': self.config.task.name,
            'total_iterations': self.iteration,
            'metric_type': self.config.metric.type,
            'initial_score': initial_score,
            'final_score': final_score,
            'improvement': improvement,
            'improvement_percent': (improvement / initial_score * 100) if initial_score > 0 else 0,
            'best_prompt': self.best_prompt,
            'initial_prompt': self.config.task.initial_prompt,
            'target_reached': self.metric_def.is_target_reached(final_score),
            'experiments_count': len(all_experiments)
        }
        
        return report
    
    def run(self) -> Dict[str, Any]:
        """Run the full optimization loop.
        
        max_iterations now represents the TOTAL number of improvement experiments
        to run, not the number of batch cycles.
        """
        logger.info("=" * 60)
        logger.info("STARTING PROMPT OPTIMIZATION")
        logger.info("=" * 60)
        
        # Generate metric if in auto mode
        if self.auto_metric:
            logger.info("Generating custom metric based on task description...")
            metric_def = self.prompt_optimizer.generate_metric()
            if metric_def:
                self.metric_def.set_custom_metric(
                    metric_def.get('metric_description', ''),
                    None  # Use default evaluator based on evaluation_type
                )
                # Update evaluator based on suggested type
                eval_type = metric_def.get('evaluation_type', 'semantic_similarity')
                if eval_type in ['accuracy', 'f1', 'exact_match', 'contains', 'semantic_similarity']:
                    self.metrics_evaluator = MetricsEvaluator(eval_type)
                    logger.info(f"Using {eval_type} evaluator for custom metric")
            else:
                logger.warning("Failed to generate custom metric, using semantic_similarity fallback")
        
        # Generate dataset - always generate fresh based on task config
        self.dataset = self.generate_dataset(force_refresh=True)
        if not self.dataset:
            logger.error("Cannot proceed without dataset")
            return {'status': 'failed', 'reason': 'No dataset'}
        
        # Use the full dataset for each experiment (batch_size = dataset size)
        # This ensures each experiment tests on all test cases
        logger.info(f"Using full dataset of {len(self.dataset)} test cases for each experiment")
        
        previous_score = 0.0
        
        # max_iterations is now the TOTAL number of experiments to run
        # Each iteration = one prompt tested on full dataset
        while self.iteration < self.config.experiment.max_iterations:
            self.iteration += 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"EXPERIMENT {self.iteration}/{self.config.experiment.max_iterations}")
            logger.info(f"{'='*60}")
            logger.info(f"Testing prompt on {len(self.dataset)} test cases")
            
            # Use full dataset for each experiment
            test_batch = self.dataset
            
            # Run experiment
            experiment = self.run_experiment(self.current_prompt, test_batch)
            
            # Check for duplicates (same prompt tested on same inputs)
            if self.ledger.is_duplicate_experiment(experiment):
                logger.warning("Duplicate experiment detected, generating new prompt...")
                # Force generation of a different prompt
                context = self.context_manager.get_context_for_optimizer(
                    self.current_prompt, experiment.mean_score
                )
                improved_prompt = self.prompt_optimizer.optimize(
                    context, self.current_prompt, experiment.mean_score,
                    self.config.metric.type
                )
                if improved_prompt and improved_prompt != self.current_prompt:
                    self.current_prompt = improved_prompt
                    logger.info("Generated alternative prompt to avoid duplicate")
                continue
            
            # Add to ledger
            self.ledger.add_experiment(experiment)
            
            # Update best if improved (with margin to avoid noise)
            current_score = experiment.mean_score
            score_margin = 0.001  # Small margin to avoid floating point noise
            if current_score > (self.best_score + score_margin):
                improvement = current_score - self.best_score
                self.best_score = current_score
                self.best_prompt = self.current_prompt
                logger.info(f"*** NEW BEST *** Score: {self.best_score:.3f} (+{improvement:.3f})")
            else:
                logger.info(f"Score: {current_score:.3f} (best: {self.best_score:.3f})")
            
            # Add to context manager
            exp_dict = {
                'iteration': experiment.iteration,
                'prompt': experiment.prompt,
                'metric_score': experiment.mean_score,
                'improvement': current_score - previous_score,
                'sample_results': [
                    {
                        'input': inp,
                        'expected': exp,
                        'actual': act,
                        'score': score
                    }
                    for inp, exp, act, score in zip(
                        experiment.inputs[:3],  # Show 3 samples for better context
                        experiment.expected_outputs[:3],
                        experiment.actual_outputs[:3],
                        experiment.metric_scores[:3]
                    )
                ]
            }
            self.context_manager.add_experiment(exp_dict)
            
            # Check convergence
            if self.check_convergence(current_score, previous_score):
                logger.info("Convergence criteria met, stopping optimization")
                break
            
            # Generate improved prompt with detailed feedback
            context = self.context_manager.get_context_for_optimizer(
                self.current_prompt, current_score
            )
            
            # Build detailed feedback summary from the experiment results
            feedback_summary = self._build_feedback_summary(experiment)
            
            improved_prompt = self.prompt_optimizer.optimize(
                context,
                self.current_prompt,
                current_score,
                self.config.metric.type,
                feedback_summary
            )
            
            if improved_prompt:
                self.current_prompt = improved_prompt
            else:
                logger.error("Failed to generate improved prompt, stopping")
                break
            
            previous_score = current_score
            
            # Save checkpoint periodically
            if self.iteration % self.config.storage.checkpoint_interval == 0:
                self.save_checkpoint()
        
        # Generate final report
        report = self.generate_summary_report()
        
        # Save report
        report_path = os.path.join(
            self.config.storage.results_dir,
            'final_report.json'
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total iterations: {report['total_iterations']}")
        logger.info(f"Initial score: {report['initial_score']:.3f}")
        logger.info(f"Final score: {report['final_score']:.3f}")
        logger.info(f"Improvement: {report['improvement']:.3f} ({report['improvement_percent']:.1f}%)")
        logger.info(f"Target reached: {'Yes' if report['target_reached'] else 'No'}")
        logger.info(f"Report saved to: {report_path}")
        logger.info("")
        
        return report


if __name__ == "__main__":
    # Test the system
    from config_manager import load_config
    
    print("Testing Optimization System...")
    
    # This would need actual API calls to test fully
    print("Optimization system module loaded successfully")
"""
Context Management Module for handling Optimizer LLM context window.
Implements strategies to compress and prioritize experiment history.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSummary:
    """Summary of an experiment for context."""
    iteration: int
    prompt: str
    score: float
    improvement: float
    key_insight: str


class ContextManager:
    """Manages context for Optimizer LLM to handle growing experiment history."""
    
    def __init__(self, max_experiments: int = 20, 
                 compression_threshold: int = 50):
        """Initialize context manager."""
        self.max_experiments = max_experiments
        self.compression_threshold = compression_threshold
        self.experiment_history: List[Dict[str, Any]] = []
        self.compressed_summary: Optional[str] = None
        self.iteration_count = 0
    
    def add_experiment(self, experiment: Dict[str, Any]):
        """Add an experiment to history."""
        self.experiment_history.append(experiment)
        self.iteration_count += 1
        
        # Compress if threshold reached
        if len(self.experiment_history) >= self.compression_threshold:
            self._compress_history()
    
    def _compress_history(self):
        """Compress older experiments into summary."""
        if len(self.experiment_history) <= self.max_experiments:
            return
        
        # Keep only recent experiments
        to_compress = self.experiment_history[:-self.max_experiments]
        self.experiment_history = self.experiment_history[-self.max_experiments:]
        
        # Create summary of compressed experiments
        summary_parts = []
        
        # Calculate statistics
        scores = [exp.get('metric_score', 0) for exp in to_compress]
        avg_score = sum(scores) / len(scores) if scores else 0
        best_score = max(scores) if scores else 0
        
        summary_parts.append(
            f"Compressed {len(to_compress)} earlier experiments: "
            f"avg_score={avg_score:.3f}, best_score={best_score:.3f}"
        )
        
        # Extract key insights from best experiments
        sorted_exps = sorted(to_compress, 
                            key=lambda x: x.get('metric_score', 0), 
                            reverse=True)
        
        top_experiments = sorted_exps[:3]
        if top_experiments:
            summary_parts.append("Top performing approaches:")
            for exp in top_experiments:
                prompt_preview = exp.get('prompt', '')[:100]
                score = exp.get('metric_score', 0)
                summary_parts.append(f"  - Score {score:.3f}: {prompt_preview}...")
        
        # Add to compressed summary
        if self.compressed_summary:
            self.compressed_summary += "\n" + "\n".join(summary_parts)
        else:
            self.compressed_summary = "\n".join(summary_parts)
        
        logger.info(f"Compressed {len(to_compress)} experiments into summary")
    
    def get_context_for_optimizer(self, current_prompt: str,
                                   current_score: float) -> str:
        """Generate context string for Optimizer LLM."""
        context_parts = []
        
        # Add compressed summary if exists
        if self.compressed_summary:
            context_parts.append("=== PREVIOUS EXPERIMENTS (SUMMARIZED) ===")
            context_parts.append(self.compressed_summary)
            context_parts.append("")
        
        # Add recent experiments
        if self.experiment_history:
            context_parts.append("=== RECENT EXPERIMENTS ===")
            
            for exp in self.experiment_history[-self.max_experiments:]:
                exp_str = self._format_experiment(exp)
                context_parts.append(exp_str)
                context_parts.append("")
        
        # Add current state
        context_parts.append("=== CURRENT STATE ===")
        context_parts.append(f"Current Prompt: {current_prompt}")
        context_parts.append(f"Current Score: {current_score:.3f}")
        context_parts.append(f"Total Iterations: {self.iteration_count}")
        
        return "\n".join(context_parts)
    
    def _format_experiment(self, exp: Dict[str, Any]) -> str:
        """Format a single experiment for context."""
        lines = [
            f"Iteration {exp.get('iteration', 'N/A')}:",
            f"  Prompt: {exp.get('prompt', 'N/A')[:200]}...",
            f"  Score: {exp.get('metric_score', 0):.3f}",
        ]
        
        # Add improvement if available
        if 'improvement' in exp:
            lines.append(f"  Improvement: {exp['improvement']:+.3f}")
        
        # Add sample results if available
        if 'sample_results' in exp and exp['sample_results']:
            lines.append("  Sample Results:")
            for i, result in enumerate(exp['sample_results'][:2], 1):
                lines.append(f"    {i}. Input: {result.get('input', 'N/A')[:50]}...")
                lines.append(f"       Expected: {result.get('expected', 'N/A')}")
                lines.append(f"       Got: {result.get('actual', 'N/A')}")
                lines.append(f"       Score: {result.get('score', 0):.2f}")
        
        return "\n".join(lines)
    
    def get_experiment_count(self) -> int:
        """Get total number of experiments."""
        return self.iteration_count
    
    def get_recent_experiments(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent experiments."""
        return self.experiment_history[-n:] if self.experiment_history else []
    
    def get_best_experiment(self) -> Optional[Dict[str, Any]]:
        """Get the experiment with highest score."""
        if not self.experiment_history:
            return None
        
        return max(self.experiment_history, 
                  key=lambda x: x.get('metric_score', 0))
    
    def clear(self):
        """Clear all history."""
        self.experiment_history = []
        self.compressed_summary = None
        self.iteration_count = 0
        logger.info("Context manager cleared")


if __name__ == "__main__":
    # Test context manager
    print("Testing Context Manager...")
    
    cm = ContextManager(max_experiments=3, compression_threshold=5)
    
    # Add some test experiments
    for i in range(7):
        exp = {
            'iteration': i + 1,
            'prompt': f'Test prompt version {i+1}',
            'metric_score': 0.5 + i * 0.05,
            'improvement': 0.05 if i > 0 else 0
        }
        cm.add_experiment(exp)
    
    print(f"\nTotal experiments: {cm.get_experiment_count()}")
    print(f"Recent experiments in memory: {len(cm.experiment_history)}")
    
    context = cm.get_context_for_optimizer("Current prompt", 0.75)
    print(f"\nContext length: {len(context)} chars")
    print("\nContext preview:")
    print(context[:500] + "...")

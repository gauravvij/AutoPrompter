"""
Prompt Optimizer Module using Optimizer LLM.
Generates improved prompts based on experiment history and results.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Optimizes prompts using an LLM based on experiment results."""
    
    def __init__(self, llm_client: LLMClient, task_config):
        """Initialize with LLM client and task configuration."""
        self.llm_client = llm_client
        self.task_config = task_config
        self.optimization_count = 0
        self.stagnation_count = 0  # Track iterations with no improvement
        self.last_score = 0.0
        self.diversify_mode = False  # Flag for diversification mode
    
    def _build_optimization_prompt(self, context: str, 
                                    current_prompt: str,
                                    current_score: float,
                                    metric_name: str,
                                    feedback_summary: str = "") -> str:
        """Build prompt for optimization with chain-of-thought requirements."""
        
        # Add diversification instruction if in diversify mode
        diversify_instruction = ""
        if self.diversify_mode:
            diversify_instruction = """

⚠️ CRITICAL - SCORES HAVE BEEN STAGNANT FOR MULTIPLE ITERATIONS ⚠️
The current approach is NOT working. You MUST try a RADICALLY DIFFERENT strategy:
- If current prompt asks for step-by-step, try a different structure (bullet points, numbered lists, or free-form)
- If current prompt is brief, try a more detailed instructional prompt
- If current prompt is detailed, try a minimal, focused prompt
- Consider adding explicit formatting instructions (e.g., "Start your answer with 'Final Answer:'")
- Try asking the model to verify its answer
- Consider role-playing approaches ("You are a logic expert...")

YOU MUST PRODUCE A MEANINGFULLY DIFFERENT PROMPT. Repeating similar prompts will fail."""
        
        # Add failure analysis for zero scores
        failure_analysis = ""
        if current_score < 0.01:
            failure_analysis = """

⚠️ CRITICAL - CURRENT SCORE IS ZERO OR NEAR-ZERO ⚠️
This indicates a FUNDAMENTAL mismatch between the expected output format and what the model is producing.

You MUST analyze:
1. What format does the expected output use? (e.g., "Step 1:", "Final Answer:", bullet points)
2. What format is the model actually producing? (e.g., markdown headers, paragraphs, different structure)
3. How can the prompt be modified to align these formats?

RECOMMENDATIONS for zero-score situations:
- Add explicit format instructions: "Structure your response as: Step 1: [reasoning], Step 2: [reasoning], Final Answer: [answer]"
- Request specific markers: "Begin your final answer with the phrase 'Final Answer:'"
- Ask for structured output: "Provide your reasoning in numbered steps, then state the final answer clearly"
- Consider that the model may need more explicit guidance on output format than content"""
        
        # Extract negative constraints from context - what has NOT worked
        negative_constraints = ""
        if context and "Previous Experiments" in context:
            # Parse failed approaches from context
            failed_approaches = []
            lines = context.split('\n')
            in_experiment = False
            current_score = 0.0
            current_prompt_snippet = ""
            
            for line in lines:
                if line.startswith('Experiment '):
                    in_experiment = True
                    # Extract score if present
                    if 'Score:' in line:
                        try:
                            score_str = line.split('Score:')[1].split()[0]
                            current_score = float(score_str)
                        except:
                            pass
                elif in_experiment and line.startswith('Prompt: '):
                    current_prompt_snippet = line[8:100]  # First 100 chars
                elif in_experiment and line.strip() == '':
                    # End of experiment section
                    if current_score < 0.5 and current_prompt_snippet:
                        failed_approaches.append(f"(Score {current_score:.2f}) {current_prompt_snippet}...")
                    in_experiment = False
                    current_score = 0.0
                    current_prompt_snippet = ""
            
            if failed_approaches:
                negative_constraints = f"""

🚫 NEGATIVE CONSTRAINTS - APPROACHES THAT HAVE FAILED:
The following prompt strategies have ALREADY BEEN TRIED and resulted in LOW SCORES.
DO NOT repeat these approaches - they do NOT work for this task:
"""
                for i, approach in enumerate(failed_approaches[:5], 1):  # Top 5 failed
                    negative_constraints += f"\n  {i}. {approach}"
                negative_constraints += "\n\nYou MUST try a DIFFERENT approach than the ones listed above."
        
        # Add detailed feedback summary if provided
        feedback_section = ""
        if feedback_summary:
            feedback_section = f"""

📊 DETAILED FEEDBACK FROM PREVIOUS ITERATION:
{feedback_summary}"""
        
        return f"""You are a prompt optimization expert. Your task is to improve a prompt based on its performance history.

Task Information:
- Name: {self.task_config.name}
- Description: {self.task_config.description}

Current Prompt:
```
{current_prompt}
```

Current Performance:
- Score: {current_score:.3f} (using {metric_name} metric)
- Iteration: {self.optimization_count}
- Stagnation Count: {self.stagnation_count}

{context}
{negative_constraints}
{failure_analysis}
{diversify_instruction}
{feedback_section}

REQUIRED - Chain-of-Thought Analysis:
Before proposing a new prompt, you MUST analyze the following:
1. **Failure Pattern Analysis**: What specific patterns do you see in the failures? Are there common format mismatches?
2. **Format Alignment**: Does the expected output use specific markers ("Step 1:", "Final Answer:") that the model isn't producing?
3. **Root Cause**: Is the issue with the prompt's clarity, the output format specification, or the reasoning instructions?
4. **Hypothesis**: What specific change do you believe will improve the score?

Your task:
1. First, provide your chain-of-thought analysis (this will not be used as the prompt)
2. Then, propose an improved prompt that addresses the identified issues
3. The new prompt should be clear, specific, and optimized for the task

Important guidelines:
- Make meaningful changes, not just cosmetic ones
- Keep the prompt concise but complete
- Consider edge cases and ambiguous inputs
- Ensure the output format is clearly specified if needed
- If scores are low, focus on format alignment between expected and actual outputs

Respond with ONLY the improved prompt text, nothing else. Do not include markdown code blocks or explanations."""
    
    def optimize(self, context: str, current_prompt: str,
                 current_score: float, metric_name: str,
                 feedback_summary: str = "") -> Optional[str]:
        """Generate an improved prompt based on context and results.
        
        Implements diversification strategy: if score remains same for 2+ iterations,
        generates 3 diverse candidates and selects the one with highest predicted potential.
        
        Args:
            context: Context from previous experiments
            current_prompt: Current prompt being optimized
            current_score: Current performance score
            metric_name: Name of the metric being used
            feedback_summary: Detailed feedback about failures (optional)
        """
        logger.info(f"Optimizing prompt (current score: {current_score:.3f})")
        
        # Track stagnation
        if abs(current_score - self.last_score) < 0.001:
            self.stagnation_count += 1
            logger.info(f"Score stagnation detected: count={self.stagnation_count}")
        else:
            self.stagnation_count = 0
            self.diversify_mode = False
        
        self.last_score = current_score
        
        # Activate diversify mode after 2 stagnant iterations
        if self.stagnation_count >= 2:
            self.diversify_mode = True
            logger.info("ACTIVATING DIVERSIFICATION MODE - will generate 3 diverse candidates")
            return self._optimize_diverse(context, current_prompt, current_score, metric_name, feedback_summary)
        
        # Standard single-prompt optimization
        return self._optimize_single(context, current_prompt, current_score, metric_name, feedback_summary)
    
    def _optimize_single(self, context: str, current_prompt: str,
                         current_score: float, metric_name: str,
                         feedback_summary: str = "") -> Optional[str]:
        """Generate a single improved prompt."""
        prompt = self._build_optimization_prompt(
            context, current_prompt, current_score, metric_name, feedback_summary
        )
        
        response = self.llm_client.query(
            prompt,
            system_message="You are a prompt optimization expert. Provide only the improved prompt text without any additional commentary or markdown formatting."
        )
        
        if not response.success:
            logger.error(f"Optimization failed: {response.error}")
            return None
        
        improved_prompt = self._clean_prompt_response(response.content)
        self.optimization_count += 1
        logger.info(f"Generated improved prompt ({len(improved_prompt)} chars)")
        
        return improved_prompt
    
    def _optimize_diverse(self, context: str, current_prompt: str,
                          current_score: float, metric_name: str,
                          feedback_summary: str = "") -> Optional[str]:
        """Generate 3 diverse prompt candidates and select the best."""
        logger.info("Generating 3 diverse prompt candidates...")
        
        candidates = []
        strategies = [
            "structured_step_by_step",
            "minimal_directive", 
            "expert_roleplay"
        ]
        
        for i, strategy in enumerate(strategies):
            # Pass feedback_summary to _build_diverse_prompt
            strategy_prompt = self._build_diverse_prompt(
                context, current_prompt, current_score, metric_name, strategy, feedback_summary
            )
            
            response = self.llm_client.query(
                strategy_prompt,
                system_message=f"You are a prompt optimization expert using the '{strategy}' strategy. Provide only the improved prompt text."
            )
            
            if response.success and response.content:
                candidate = self._clean_prompt_response(response.content)
                candidates.append((candidate, strategy))
                logger.info(f"Candidate {i+1} ({strategy}): {len(candidate)} chars")
            else:
                logger.warning(f"Failed to generate candidate {i+1}")
        
        if not candidates:
            logger.error("All diversification candidates failed, falling back to single optimization")
            return self._optimize_single(context, current_prompt, current_score, metric_name, feedback_summary)
        
        # Select the best candidate based on predicted potential
        best_candidate = self._select_best_candidate(candidates, context, current_score)
        
        self.optimization_count += 1
        self.diversify_mode = False  # Reset after diversification
        self.stagnation_count = 0
        
        logger.info(f"Selected best diverse prompt ({len(best_candidate)} chars)")
        return best_candidate
    
    def _build_diverse_prompt(self, context: str, current_prompt: str,
                              current_score: float, metric_name: str,
                              strategy: str, feedback_summary: str = "") -> str:
        """Build a diversification prompt for a specific strategy."""
        
        strategy_instructions = {
            "structured_step_by_step": """Use a highly structured approach:
- Explicitly request numbered steps (Step 1, Step 2, etc.)
- Ask for clear section headers
- Request specific formatting like "Reasoning:" and "Conclusion:"
- Emphasize logical progression""",
            
            "minimal_directive": """Use a minimal, focused approach:
- Remove all unnecessary instructions
- Keep only the core task description
- Let the model's inherent reasoning abilities work
- Avoid over-specifying format
- Trust the model to produce quality output""",
            
            "expert_roleplay": """Use an expert roleplay approach:
- Frame the task as if consulting an expert ("You are a math professor...")
- Ask the model to explain as if teaching a student
- Request verification of each step
- Emphasize precision and correctness"""
        }
        
        feedback_section = f"""

📊 DETAILED FEEDBACK FROM PREVIOUS ITERATION:
{feedback_summary}""" if feedback_summary else ""
        
        return f"""You are a prompt optimization expert. Create a RADICALLY DIFFERENT prompt using the '{strategy}' strategy.

Task Information:
- Name: {self.task_config.name}
- Description: {self.task_config.description}

Current Prompt (which is NOT working - score: {current_score:.3f}):
```
{current_prompt}
```

Strategy to use:
{strategy_instructions.get(strategy, "Create a different approach")}

{context}

IMPORTANT: Your new prompt MUST be significantly different from the current one. Do NOT just rephrase - use a completely different structure and approach.

{feedback_section}

Respond with ONLY the new prompt text, nothing else."""
    
    def _select_best_candidate(self, candidates: List[Tuple[str, str]], 
                                context: str, current_score: float) -> str:
        """Select the best prompt candidate based on predicted potential.
        
        Uses the Optimizer LLM to evaluate which candidate has the highest
        predicted potential for success.
        """
        if len(candidates) == 1:
            return candidates[0][0]
        
        logger.info(f"Evaluating {len(candidates)} diverse candidates...")
        
        # Build evaluation prompt
        eval_prompt = f"""You are a prompt evaluation expert. Review these {len(candidates)} diverse prompt candidates and select the one with the HIGHEST potential to improve performance.

Task: {self.task_config.name}
Description: {self.task_config.description}

Current Score: {current_score:.3f}

{context}

CANDIDATE PROMPTS:
"""
        
        for i, (candidate, strategy) in enumerate(candidates, 1):
            eval_prompt += f"\n--- CANDIDATE {i} ({strategy}) ---\n{candidate}\n"
        
        eval_prompt += """

Your task:
1. Analyze each candidate's strengths and weaknesses
2. Consider which approach is most likely to break out of the current stagnation
3. Select the candidate number (1, 2, or 3) with the highest predicted potential

Respond with ONLY the candidate number (1, 2, or 3). No explanation needed."""
        
        response = self.llm_client.query(
            eval_prompt,
            system_message="You are a prompt evaluation expert. Respond with only the candidate number (1, 2, or 3)."
        )
        
        if response.success and response.content:
            content = response.content.strip()
            # Extract number from response
            for i, char in enumerate(content):
                if char in '123' and i < len(content) - 1 and not content[i+1].isdigit():
                    idx = int(char) - 1
                    if 0 <= idx < len(candidates):
                        logger.info(f"Selected candidate {char} ({candidates[idx][1]})")
                        return candidates[idx][0]
        
        # Fallback to first candidate if parsing fails
        logger.warning("Could not parse candidate selection, using first candidate")
        return candidates[0][0]
    
    def _clean_prompt_response(self, content: str) -> str:
        """Clean up the prompt response by removing markdown and whitespace."""
        improved_prompt = content.strip()
        
        # Remove markdown code blocks if present
        if improved_prompt.startswith("```"):
            lines = improved_prompt.split("\n")
            # Remove first line (```language)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            improved_prompt = "\n".join(lines).strip()
        
        # Check if prompt actually changed
        if improved_prompt == content:
            logger.debug("Prompt unchanged after cleaning")
        
        return improved_prompt
    
    def generate_metric_prompt(self) -> str:
        """Generate prompt for metric definition based on task."""
        return f"""You are a metric design expert. Define an appropriate evaluation metric for the following task:

Task Name: {self.task_config.name}
Task Description: {self.task_config.description}

Your response should include:
1. Metric Name: A concise name for the metric
2. Metric Description: How to evaluate responses (2-3 sentences)
3. Scoring Guidelines: 
   - Score 1.0: Perfect response criteria
   - Score 0.5: Partial response criteria  
   - Score 0.0: Incorrect/missing response criteria
4. Evaluation Type: Choose from [exact_match, contains, semantic_similarity, f1, accuracy]

Format your response as a JSON object with these fields:
{{
  "metric_name": "string",
  "metric_description": "string",
  "scoring_guidelines": {{
    "perfect": "string",
    "partial": "string",
    "incorrect": "string"
  }},
  "evaluation_type": "string"
}}

Provide ONLY the JSON response, no additional commentary."""
    
    def generate_metric(self) -> Optional[Dict[str, Any]]:
        """Generate a custom metric definition using the Optimizer LLM."""
        logger.info("Generating custom metric definition for task...")
        
        prompt = self.generate_metric_prompt()
        
        response = self.llm_client.query(
            prompt,
            system_message="You are a metric design expert. Provide only valid JSON without markdown formatting."
        )
        
        if not response.success:
            logger.error(f"Metric generation failed: {response.error}")
            return None
        
        try:
            # Parse JSON response
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            metric_def = json.loads(content)
            
            # Validate required fields
            required_fields = ['metric_name', 'metric_description', 'scoring_guidelines', 'evaluation_type']
            for field in required_fields:
                if field not in metric_def:
                    logger.warning(f"Missing field in metric definition: {field}")
            
            logger.info(f"Generated metric: {metric_def.get('metric_name', 'unnamed')}")
            logger.info(f"Evaluation type: {metric_def.get('evaluation_type', 'unknown')}")
            
            return metric_def
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metric definition: {e}")
            logger.debug(f"Response content: {response.content}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing metric: {e}")
            return None
    
    def generate_initial_dataset_prompt(self, num_samples: int) -> str:
        """Generate prompt for initial dataset generation."""
        return f"""Generate {num_samples} diverse test cases for the following task:

Task: {self.task_config.name}
Description: {self.task_config.description}

For each test case, provide:
1. An input that would be given to an AI model
2. The expected/correct output

Format as JSON array with objects containing "input" and "expected_output" fields.
Ensure diversity in the test cases and include edge cases."""


if __name__ == "__main__":
    # Test prompt optimizer
    from config_manager import Config, LLMConfig, TaskConfig
    
    print("Testing Prompt Optimizer...")
    
    config = Config(
        optimizer_llm=LLMConfig(model="qwen/qwen3-8b:free"),
        target_llm=LLMConfig(model="qwen/qwen3-8b:free"),
        experiment=None,
        task=TaskConfig(
            name="sentiment_analysis",
            description="Classify text sentiment"
        ),
        metric=None,
        context=None,
        storage=None
    )
    
    # This would need actual API calls to test fully
    print("Prompt optimizer initialized successfully")
    print(f"Task: {config.task.name}")
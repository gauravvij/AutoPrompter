"""
Metrics Module for evaluating Target LLM performance.
Supports multiple metric types including accuracy, F1, exact match, and semantic similarity.
"""

import re
import logging
from typing import Callable, Dict, Any, List, Tuple
from difflib import SequenceMatcher
import numpy as np

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """Evaluates model outputs against expected outputs using various metrics."""
    
    def __init__(self, metric_type: str = "accuracy"):
        """Initialize with metric type."""
        self.metric_type = metric_type
        self.evaluator = self._get_evaluator()
    
    def _get_evaluator(self) -> Callable[[str, str], float]:
        """Get the appropriate evaluation function."""
        evaluators = {
            'accuracy': self._accuracy,
            'f1': self._f1_score,
            'exact_match': self._exact_match,
            'contains': self._contains,
            'semantic_similarity': self._semantic_similarity
        }
        
        if self.metric_type not in evaluators:
            raise ValueError(f"Unknown metric type: {self.metric_type}")
        
        return evaluators[self.metric_type]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation for more lenient matching
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _accuracy(self, predicted: str, expected: str) -> float:
        """Calculate accuracy (exact match after normalization)."""
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)
        
        # Empty prediction is always a failure
        if not pred_norm:
            return 0.0
        
        # Check for exact match
        if pred_norm == exp_norm:
            return 1.0
        
        # Check if expected is contained in predicted (for classification tasks)
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return 1.0
        
        return 0.0
    
    def _exact_match(self, predicted: str, expected: str) -> float:
        """Calculate exact match (case-insensitive)."""
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)
        return 1.0 if pred_norm == exp_norm else 0.0
    
    def _extract_math_answer(self, text: str) -> str:
        """Extract the final numerical answer from math problem output.
        
        Looks for patterns like:
        - "Final Answer: 47"
        - "Final Answer: 47 muffins"
        - "The answer is 47"
        - Just "47" at the end
        """
        text = text.strip()
        
        # Pattern 1: "Final Answer: X" or "Final Answer: X unit"
        final_answer_match = re.search(r'final answer[:\s]+([\d\s.,/]+(?:\s*[a-zA-Z]+)?)', text, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # Pattern 2: "The answer is X"
        answer_is_match = re.search(r'(?:the\s+)?answer\s+(?:is|[:=])\s+([\d\s.,/]+(?:\s*[a-zA-Z]+)?)', text, re.IGNORECASE)
        if answer_is_match:
            return answer_is_match.group(1).strip()
        
        # Pattern 3: Look for standalone number at the end (last line or last few chars)
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line:
                # Look for number at the end of line
                num_match = re.search(r'([\d\s.,/]+(?:\s*[a-zA-Z]+)?)\s*$', line)
                if num_match:
                    candidate = num_match.group(1).strip()
                    # Ensure it's actually a number
                    if re.search(r'\d', candidate):
                        return candidate
        
        # Pattern 4: Just extract any standalone number as fallback
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            return numbers[-1]  # Return last number found
        
        return text  # Return full text if no pattern matches
    
    def _normalize_number(self, text: str) -> str:
        """Normalize a number for comparison.
        
        Handles:
        - "47" == "47 muffins"
        - "2.4" == "2.40"
        - "120 mph" == "120"
        - "129.60" == "129.6"
        """
        text = text.lower().strip()
        
        # Remove units (common math units)
        units = ['muffins', 'mph', 'miles', 'hours', 'liters', 'dollars', 'shirt', 'shirts', 
                 'hour', 'mile', 'liter', 'dollar', 'minute', 'minutes']
        for unit in units:
            text = re.sub(r'\s*' + unit + r'\s*', '', text)
        
        # Normalize decimal numbers (2.40 -> 2.4, 129.60 -> 129.6)
        text = text.strip()
        
        # Try to parse as float and format consistently
        try:
            num = float(text)
            # If it's an integer value, return as int string
            if num == int(num):
                return str(int(num))
            # Otherwise normalize decimal representation
            return str(num).rstrip('0').rstrip('.')
        except ValueError:
            pass
        
        return text.strip()
    
    def _contains(self, predicted: str, expected: str) -> float:
        """Check if expected content is contained in prediction with granular feedback.
        
        For math problems, this extracts and compares the final numerical answers
        rather than requiring the full step-by-step solution to be contained.
        """
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)
        
        # Check for exact containment first (for non-math tasks)
        if exp_norm in pred_norm:
            return 1.0
        
        # For math problems: extract and compare final answers
        pred_answer = self._extract_math_answer(predicted)
        exp_answer = self._extract_math_answer(expected)
        
        # Normalize both answers for comparison
        pred_answer_norm = self._normalize_number(pred_answer)
        exp_answer_norm = self._normalize_number(exp_answer)
        
        # Compare normalized answers
        if pred_answer_norm == exp_answer_norm:
            return 1.0
        
        # Check if the expected answer appears anywhere in the prediction
        if exp_answer_norm in pred_norm:
            return 1.0
        
        # Check for partial containment (key phrases) as fallback
        exp_tokens = set(exp_norm.split())
        pred_tokens = set(pred_norm.split())
        
        if exp_tokens and pred_tokens:
            overlap = len(exp_tokens & pred_tokens)
            coverage = overlap / len(exp_tokens)
            
            # Partial credit based on token overlap
            if coverage >= 0.7:
                return 0.5
            elif coverage >= 0.4:
                return 0.3
            elif coverage >= 0.2:
                return 0.1
        
        return 0.0
    
    def get_feedback(self, predicted: str, expected: str) -> Dict[str, Any]:
        """Get detailed feedback about why a prediction failed or succeeded."""
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)
        
        feedback = {
            'score': self.evaluate(predicted, expected),
            'predicted_length': len(predicted),
            'expected_length': len(expected),
            'issues': []
        }
        
        # Check for empty prediction
        if not pred_norm:
            feedback['issues'].append("Empty or whitespace-only prediction")
            return feedback
        
        # Check format issues
        if 'step 1' in exp_norm and 'step 1' not in pred_norm:
            feedback['issues'].append("Missing 'Step 1' formatting - expected output uses numbered steps")
        
        if 'final answer' in exp_norm and 'final answer' not in pred_norm:
            feedback['issues'].append("Missing 'Final Answer' marker - expected output has explicit final answer section")
        
        # Check for key content overlap
        exp_tokens = set(exp_norm.split())
        pred_tokens = set(pred_norm.split())
        
        if exp_tokens:
            overlap = exp_tokens & pred_tokens
            missing = exp_tokens - pred_tokens
            coverage = len(overlap) / len(exp_tokens)
            
            feedback['token_coverage'] = coverage
            feedback['missing_key_terms'] = list(missing)[:10]  # Top 10 missing terms
            
            if coverage < 0.5:
                feedback['issues'].append(f"Low content overlap ({coverage:.1%}) - prediction missing key concepts from expected output")
        
        # Check length disparity
        if len(predicted) < len(expected) * 0.3:
            feedback['issues'].append("Prediction significantly shorter than expected - may be incomplete")
        elif len(predicted) > len(expected) * 3:
            feedback['issues'].append("Prediction significantly longer than expected - may contain unnecessary content")
        
        return feedback
    
    def _f1_score(self, predicted: str, expected: str) -> float:
        """Calculate F1 score based on token overlap."""
        pred_tokens = set(self._normalize_text(predicted).split())
        exp_tokens = set(self._normalize_text(expected).split())
        
        if not pred_tokens and not exp_tokens:
            return 1.0
        
        if not pred_tokens or not exp_tokens:
            return 0.0
        
        intersection = pred_tokens & exp_tokens
        
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(exp_tokens) if exp_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _semantic_similarity(self, predicted: str, expected: str) -> float:
        """Calculate semantic similarity using sequence matcher."""
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, pred_norm, exp_norm).ratio()
        return similarity
    
    def evaluate(self, predicted: str, expected: str) -> float:
        """Evaluate a single prediction against expected output."""
        try:
            score = self.evaluator(predicted, expected)
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return 0.0
    
    def evaluate_with_feedback(self, predicted: str, expected: str) -> Tuple[float, Dict[str, Any]]:
        """Evaluate and return detailed feedback about the prediction."""
        score = self.evaluate(predicted, expected)
        feedback = self.get_feedback(predicted, expected)
        return score, feedback
    
    def evaluate_batch(self, predictions: List[str], 
                       expected: List[str]) -> Dict[str, Any]:
        """Evaluate a batch of predictions."""
        if len(predictions) != len(expected):
            raise ValueError("Predictions and expected lists must have same length")
        
        scores = []
        for pred, exp in zip(predictions, expected):
            score = self.evaluate(pred, exp)
            scores.append(score)
        
        scores_array = np.array(scores)
        
        return {
            'scores': scores,
            'mean': float(np.mean(scores_array)),
            'median': float(np.median(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'count': len(scores)
        }
    
    def get_metric_name(self) -> str:
        """Get the name of the metric."""
        return self.metric_type


class MetricDefinition:
    """Defines a metric for the optimization process."""
    
    def __init__(self, metric_type: str, target_score: float = 0.95,
                 custom_evaluator: Callable = None,
                 metric_description: str = None):
        """Initialize metric definition.
        
        Args:
            metric_type: Type of metric ('accuracy', 'f1', 'exact_match', etc. or 'auto')
            target_score: Target score to reach
            custom_evaluator: Custom evaluation function
            metric_description: Description of how to evaluate (for auto-generated metrics)
        """
        self.metric_type = metric_type
        self.target_score = target_score
        self.custom_evaluator = custom_evaluator
        self.metric_description = metric_description
        
        # Initialize evaluator if not auto mode
        if metric_type != 'auto':
            self.evaluator = MetricsEvaluator(metric_type)
        else:
            self.evaluator = None
    
    def set_custom_metric(self, description: str, evaluator_func: Callable = None):
        """Set a custom metric based on optimizer-generated description.
        
        Args:
            description: Description of how to evaluate responses
            evaluator_func: Optional custom evaluation function
        """
        self.metric_description = description
        if evaluator_func:
            self.custom_evaluator = evaluator_func
        self.evaluator = MetricsEvaluator('semantic_similarity')  # Fallback for auto mode
        logger.info(f"Custom metric set: {description[:100]}...")
    
    def evaluate(self, predicted: str, expected: str) -> float:
        """Evaluate using the defined metric."""
        if self.custom_evaluator:
            return self.custom_evaluator(predicted, expected)
        if self.evaluator:
            return self.evaluator.evaluate(predicted, expected)
        # Fallback: semantic similarity
        return MetricsEvaluator('semantic_similarity').evaluate(predicted, expected)
    
    def is_target_reached(self, score: float) -> bool:
        """Check if target score is reached."""
        return score >= self.target_score


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    test_cases = [
        ("positive", "positive", 1.0),
        ("Positive!", "positive", 1.0),
        ("The sentiment is positive", "positive", 1.0),
        ("negative", "positive", 0.0),
        ("I feel good", "positive", 0.0),
    ]
    
    for metric_type in ['accuracy', 'exact_match', 'contains', 'f1', 'semantic_similarity']:
        print(f"\n{metric_type.upper()}:")
        evaluator = MetricsEvaluator(metric_type)
        
        for pred, exp, _ in test_cases:
            score = evaluator.evaluate(pred, exp)
            print(f"  '{pred}' vs '{exp}': {score:.3f}")

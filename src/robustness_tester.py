"""
Robustness Testing Module for AutoPrompter.

Generates perturbed variants of test inputs to ensure prompts generalize
across different input variations (typos, paraphrasing, format variations,
ambiguous phrasing).
"""

import random
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dataset_generator import DatasetEntry

logger = logging.getLogger(__name__)


@dataclass
class RobustnessResult:
    """Result of robustness testing for a single prompt."""
    original_score: float
    robustness_score: float  # Average score across all variants
    score_variance: float
    num_variants: int
    variant_scores: List[Dict[str, Any]]
    passed: bool  # True if robustness_score >= threshold


class RobustnessTester:
    """Tests prompt robustness by generating input perturbations."""
    
    # Common typo patterns for keyboard proximity
    KEYBOARD_NEIGHBORS = {
        'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx',
        'e': 'wrsd', 'f': 'dcvgtr', 'g': 'fvbhty', 'h': 'gbnjyu',
        'i': 'ujko', 'j': 'hnmkiu', 'k': 'jmloi', 'l': 'kop',
        'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
        'q': 'wa', 'r': 'etdf', 's': 'wadxz', 't': 'ryfg',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
        'y': 'tghu', 'z': 'asx'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize robustness tester with configuration.
        
        Args:
            config: Configuration dict with keys:
                - enabled: bool (default True)
                - num_variants: int (default 3)
                - score_threshold: float (default 0.9)
                - strategies: List[str] (default all)
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.num_variants = self.config.get('num_variants', 3)
        self.score_threshold = self.config.get('score_threshold', 0.9)
        self.strategies = self.config.get('strategies', [
            'typos', 'paraphrase', 'format_variation', 'ambiguous'
        ])
        
        logger.info(f"RobustnessTester initialized: enabled={self.enabled}, "
                   f"variants={self.num_variants}, strategies={self.strategies}")
    
    def generate_variants(self, entry: DatasetEntry) -> List[DatasetEntry]:
        """Generate perturbed variants of a dataset entry.
        
        Args:
            entry: Original dataset entry
            
        Returns:
            List of variant entries with perturbations applied
        """
        if not self.enabled:
            return [entry]
        
        variants = [entry]  # Include original
        
        for strategy in self.strategies:
            for i in range(self.num_variants // len(self.strategies) + 1):
                if len(variants) >= self.num_variants + 1:
                    break
                    
                variant_input = self._apply_perturbation(entry.input, strategy)
                if variant_input != entry.input:
                    variant = DatasetEntry(
                        input=variant_input,
                        expected_output=entry.expected_output,
                        metadata={
                            **(entry.metadata or {}),
                            'perturbation_strategy': strategy,
                            'perturbation_index': i,
                            'original_input': entry.input
                        }
                    )
                    variants.append(variant)
        
        return variants[:self.num_variants + 1]
    
    def _apply_perturbation(self, text: str, strategy: str) -> str:
        """Apply a specific perturbation strategy to text.
        
        Args:
            text: Original text
            strategy: Perturbation strategy to apply
            
        Returns:
            Perturbed text
        """
        if strategy == 'typos':
            return self._add_typos(text)
        elif strategy == 'paraphrase':
            return self._paraphrase(text)
        elif strategy == 'format_variation':
            return self._vary_format(text)
        elif strategy == 'ambiguous':
            return self._add_ambiguity(text)
        else:
            return text
    
    def _add_typos(self, text: str, typo_rate: float = 0.05) -> str:
        """Add realistic typos to text based on keyboard proximity.
        
        Args:
            text: Original text
            typo_rate: Probability of typo per character
            
        Returns:
            Text with typos added
        """
        chars = list(text)
        num_typos = max(1, int(len(chars) * typo_rate))
        
        for _ in range(num_typos):
            if len(chars) < 2:
                break
            
            idx = random.randint(0, len(chars) - 1)
            char = chars[idx].lower()
            
            if char in self.KEYBOARD_NEIGHBORS:
                neighbors = self.KEYBOARD_NEIGHBORS[char]
                chars[idx] = random.choice(neighbors)
        
        return ''.join(chars)
    
    def _paraphrase(self, text: str) -> str:
        """Create a paraphrased version of the text.
        
        Uses simple synonym replacement and sentence restructuring.
        
        Args:
            text: Original text
            
        Returns:
            Paraphrased text
        """
        # Simple synonym replacements
        synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['poor', 'terrible', 'awful', 'unfavorable'],
            'big': ['large', 'huge', 'massive', 'substantial'],
            'small': ['tiny', 'little', 'minor', 'slight'],
            'important': ['significant', 'crucial', 'essential', 'vital'],
            'think': ['believe', 'consider', 'suppose', 'assume'],
            'use': ['utilize', 'employ', 'apply', 'leverage'],
            'make': ['create', 'produce', 'generate', 'build'],
            'get': ['obtain', 'acquire', 'receive', 'gain'],
            'show': ['display', 'demonstrate', 'exhibit', 'reveal']
        }
        
        words = text.split()
        paraphrased = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in synonyms and random.random() < 0.3:
                replacement = random.choice(synonyms[clean_word])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                paraphrased.append(replacement + word[len(clean_word):])
            else:
                paraphrased.append(word)
        
        return ' '.join(paraphrased)
    
    def _vary_format(self, text: str) -> str:
        """Apply format variations to text.
        
        Args:
            text: Original text
            
        Returns:
            Text with format variations
        """
        variations = [
            lambda t: t.upper(),
            lambda t: t.lower(),
            lambda t: t.capitalize(),
            lambda t: t.replace('?', ' ??').replace('!', ' !!'),
            lambda t: t.replace('.', ' .').replace(',', ' ,'),
            lambda t: ' '.join(t.split()),  # Normalize whitespace
            lambda t: t.replace('and', '&').replace('or', '/'),
        ]
        
        # Apply 1-2 random variations
        result = text
        for _ in range(random.randint(1, 2)):
            variation = random.choice(variations)
            try:
                result = variation(result)
            except:
                pass
        
        return result
    
    def _add_ambiguity(self, text: str) -> str:
        """Add ambiguous phrasing to text.
        
        Args:
            text: Original text
            
        Returns:
            Text with ambiguous elements
        """
        ambiguous_prefixes = [
            "I'm not sure, but ",
            "Maybe ",
            "Possibly ",
            "It seems like ",
            "I think ",
            "Perhaps ",
            "It might be that ",
        ]
        
        ambiguous_suffixes = [
            " or something",
            " maybe",
            " I guess",
            " probably",
            " or whatever",
        ]
        
        result = text
        
        # Add prefix with 30% probability
        if random.random() < 0.3:
            result = random.choice(ambiguous_prefixes) + result[0].lower() + result[1:]
        
        # Add suffix with 30% probability
        if random.random() < 0.3:
            result = result.rstrip('.!?') + random.choice(ambiguous_suffixes)
        
        return result
    
    def compute_robustness_score(self, original_score: float,
                                  variant_scores: List[float]) -> RobustnessResult:
        """Compute robustness score from original and variant scores.
        
        Args:
            original_score: Score on original inputs
            variant_scores: Scores on perturbed variants
            
        Returns:
            RobustnessResult with aggregated metrics
        """
        all_scores = [original_score] + variant_scores
        
        robustness_score = sum(all_scores) / len(all_scores)
        score_variance = sum((s - robustness_score) ** 2 for s in all_scores) / len(all_scores)
        
        variant_details = [
            {
                'variant_index': i,
                'score': score,
                'drop_from_original': original_score - score
            }
            for i, score in enumerate(variant_scores)
        ]
        
        passed = robustness_score >= (original_score * self.score_threshold)
        
        return RobustnessResult(
            original_score=original_score,
            robustness_score=robustness_score,
            score_variance=score_variance,
            num_variants=len(variant_scores),
            variant_scores=variant_details,
            passed=passed
        )
    
    def should_retry_with_robustness(self, result: RobustnessResult) -> bool:
        """Determine if prompt should be retried due to robustness failure.
        
        Args:
            result: Robustness test result
            
        Returns:
            True if prompt failed robustness and should be retried
        """
        if not self.enabled:
            return False
        
        # Retry if robustness score is significantly lower than original
        score_drop = result.original_score - result.robustness_score
        return score_drop > 0.15 or not result.passed


if __name__ == "__main__":
    # Test robustness tester
    tester = RobustnessTester()
    
    test_entry = DatasetEntry(
        input="This is a test sentence for classification.",
        expected_output="positive"
    )
    
    variants = tester.generate_variants(test_entry)
    print(f"Generated {len(variants)} variants:")
    for i, v in enumerate(variants):
        print(f"  {i}: {v.input}")
        if v.metadata:
            print(f"     Strategy: {v.metadata.get('perturbation_strategy', 'original')}")

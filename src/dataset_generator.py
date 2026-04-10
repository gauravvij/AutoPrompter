"""
Dataset Generation Module using Optimizer LLM.
Generates synthetic input-output pairs relevant to the configured task.
"""

import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class DatasetEntry:
    """Single dataset entry with input and expected output."""
    input: str
    expected_output: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetEntry':
        return cls(**data)


class DatasetGenerator:
    """Generates synthetic datasets using Optimizer LLM."""

    def __init__(self, llm_client: LLMClient, task_config):
        self.llm_client = llm_client
        self.task_config = task_config

    # ------------------------------------------------------------------ #
    #  JSON parsing — robust multi-strategy                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fix_json(text: str) -> str:
        """Apply lightweight fixes for common LLM JSON mistakes."""
        # Remove trailing commas before ] or }
        text = re.sub(r',\s*([}\]])', r'\1', text)
        # Remove JS-style // comments
        text = re.sub(r'//[^\n]*', '', text)
        # Remove /* … */ comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text

    @staticmethod
    def _extract_json_array(text: str) -> Optional[list]:
        """Extract the outermost JSON array from arbitrary text."""
        start = text.find('[')
        if start == -1:
            return None
        # Walk forward counting brackets to find the matching ]
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(DatasetGenerator._fix_json(candidate))
                        except json.JSONDecodeError:
                            pass
        return None

    @staticmethod
    def _extract_objects(text: str) -> list:
        """Regex-extract individual JSON objects when the array wrapper is broken."""
        results = []
        # Match {...} blocks (non-nested for simplicity)
        for m in re.finditer(r'\{[^{}]+\}', text, re.DOTALL):
            raw = m.group(0)
            for attempt in (raw, DatasetGenerator._fix_json(raw)):
                try:
                    obj = json.loads(attempt)
                    if isinstance(obj, dict):
                        results.append(obj)
                        break
                except json.JSONDecodeError:
                    pass
        return results

    def _robust_parse(self, text: str) -> list:
        """Try every strategy to get a list of dicts from LLM output."""
        text = text.strip()

        # Strip markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # 1. Direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 2. Direct parse after fixes
        try:
            data = json.loads(self._fix_json(text))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 3. Extract outermost array
        arr = self._extract_json_array(text)
        if arr is not None:
            return arr

        # 4. Pull individual objects
        objs = self._extract_objects(text)
        if objs:
            return objs

        return []

    def _parse_dataset_response(self, response: str) -> List[DatasetEntry]:
        items = self._robust_parse(response)
        entries = []
        for item in items:
            if isinstance(item, dict) and 'input' in item and 'expected_output' in item:
                inp = str(item['input']).strip()
                out = str(item['expected_output']).strip()
                if inp and out:
                    entries.append(DatasetEntry(
                        input=inp,
                        expected_output=out,
                        metadata=item.get('metadata', {})
                    ))
        return entries

    # ------------------------------------------------------------------ #
    #  Prompt builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_generation_prompt(self, num_samples: int) -> str:
        return f"""Generate exactly {num_samples} test cases for this task as a JSON array.

Task: {self.task_config.description}

Return ONLY valid JSON — no explanation, no markdown, no extra text.
Format:
[
  {{"input": "...", "expected_output": "..."}},
  {{"input": "...", "expected_output": "..."}}
]

Rules:
- Use only plain ASCII characters
- Keep each input under 200 characters
- Keep each expected_output under 50 characters
- Make inputs diverse and realistic

Generate {num_samples} entries now:"""

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def _create_fallback_entries(self, num_samples: int) -> List[DatasetEntry]:
        """Hardcoded fallback so the run is never blocked by dataset failure."""
        task_name = self.task_config.name.lower()
        desc = self.task_config.description.lower()

        if any(w in task_name + desc for w in ('sentiment', 'positive', 'negative', 'classify')):
            pool = [
                ("I absolutely love this product!", "positive"),
                ("This is the worst experience I have ever had.", "negative"),
                ("The service was fantastic and staff were helpful.", "positive"),
                ("I am very disappointed with this purchase.", "negative"),
                ("Everything worked perfectly, highly recommend!", "positive"),
                ("Terrible quality, broke after one day.", "negative"),
                ("Surprisingly good value for money.", "positive"),
                ("Would not recommend this to anyone.", "negative"),
                ("Great customer support, resolved my issue fast.", "positive"),
                ("The product description was completely misleading.", "negative"),
            ]
        elif any(w in task_name + desc for w in ('translat', 'language')):
            pool = [
                ("Hello, how are you?", "Bonjour, comment allez-vous?"),
                ("Good morning!", "Bonjour!"),
                ("Thank you very much.", "Merci beaucoup."),
                ("Where is the station?", "Où est la gare?"),
                ("I need help.", "J'ai besoin d'aide."),
            ]
        elif any(w in task_name + desc for w in ('summar', 'abstract')):
            pool = [
                ("The quick brown fox jumps over the lazy dog. Dogs are common pets.", "Fox jumps over dog."),
                ("Python is a programming language. It is widely used in data science.", "Python is popular in data science."),
                ("The meeting was held on Monday. All teams attended and discussed Q3 goals.", "Monday meeting discussed Q3 goals."),
                ("Climate change is affecting global weather patterns significantly.", "Climate change alters weather patterns."),
                ("The company reported record profits last quarter due to strong sales.", "Company had record profits from strong sales."),
            ]
        else:
            pool = [
                (f"Sample input {i + 1} for {self.task_config.name}",
                 f"Expected output {i + 1}")
                for i in range(10)
            ]

        entries = []
        for i in range(num_samples):
            inp, out = pool[i % len(pool)]
            suffix = f" [{i // len(pool) + 1}]" if i >= len(pool) else ""
            entries.append(DatasetEntry(
                input=inp + suffix,
                expected_output=out,
                metadata={"source": "fallback"}
            ))
        logger.warning(f"Using {num_samples} hardcoded fallback entries for '{self.task_config.name}'")
        return entries

    def generate(self, num_samples: int, max_retries: int = 2) -> List[DatasetEntry]:
        logger.info(f"Generating {num_samples} dataset samples...")

        for attempt in range(max_retries):
            prompt = self._build_generation_prompt(num_samples)
            response = self.llm_client.query(
                prompt,
                system_message="You are a JSON generator. Output only valid JSON arrays. No explanation."
            )

            if not response.success:
                logger.warning(f"Dataset generation attempt {attempt + 1} API error: {response.error}")
                continue

            entries = self._parse_dataset_response(response.content)

            if len(entries) >= max(1, num_samples // 2):
                logger.info(f"Dataset generated: {len(entries)} entries (attempt {attempt + 1})")
                return entries[:num_samples]

            logger.warning(f"Attempt {attempt + 1}: parsed {len(entries)}/{num_samples} entries, retrying...")
            if response.content:
                logger.debug(f"Raw response snippet: {response.content[:300]}")

        # All LLM attempts failed — use hardcoded fallback immediately
        logger.warning("LLM dataset generation failed, using hardcoded fallback entries")
        return self._create_fallback_entries(num_samples)

    # ------------------------------------------------------------------ #
    #  Validation / persistence                                            #
    # ------------------------------------------------------------------ #

    def validate_dataset(self, entries: List[DatasetEntry]) -> Tuple[bool, str]:
        if not entries:
            return False, "Dataset is empty"
        inputs_seen = set()
        duplicates = sum(1 for e in entries if e.input in inputs_seen or inputs_seen.add(e.input))
        if duplicates > len(entries) * 0.1:
            return False, f"Too many duplicates: {duplicates}/{len(entries)}"
        empty = sum(1 for e in entries if not e.input or not e.expected_output)
        if empty > 0:
            return False, f"Found {empty} empty entries"
        return True, f"Dataset valid: {len(entries)} entries"

    def save_dataset(self, entries: List[DatasetEntry], filepath: str):
        with open(filepath, 'w') as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)
        logger.info(f"Dataset saved to {filepath}")

    def load_dataset(self, filepath: str) -> List[DatasetEntry]:
        with open(filepath, 'r') as f:
            data = json.load(f)
        entries = [DatasetEntry.from_dict(item) for item in data]
        logger.info(f"Loaded {len(entries)} entries from {filepath}")
        return entries

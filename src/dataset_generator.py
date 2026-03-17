"""
Dataset Generation Module using Optimizer LLM.
Generates synthetic input-output pairs relevant to the configured task.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
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
        """Initialize with LLM client and task configuration."""
        self.llm_client = llm_client
        self.task_config = task_config
    
    def _build_generation_prompt(self, num_samples: int) -> str:
        """Build prompt for dataset generation."""
        return f"""You are a dataset generation assistant. Generate {num_samples} diverse and realistic test cases for the following task:

Task Name: {self.task_config.name}
Task Description: {self.task_config.description}

For each test case, provide:
1. An input that would be given to an AI model
2. The expected/correct output

Requirements:
- Ensure inputs are diverse and cover different scenarios
- Make inputs realistic and contextually relevant
- Expected outputs should be accurate and consistent
- Include edge cases and challenging examples
- Format as a JSON array with objects containing "input" and "expected_output" fields
- IMPORTANT: Use simple ASCII characters only, avoid special quotes or unicode
- Keep inputs and outputs concise (under 500 chars each)

Example format:
[
  {{
    "input": "example input text",
    "expected_output": "example expected response"
  }}
]

Generate exactly {num_samples} test cases."""
    
    def _parse_dataset_response(self, response: str) -> List[DatasetEntry]:
        """Parse LLM response into dataset entries with robust error handling."""
        try:
            # Try to extract JSON from response
            content = response.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Try to find JSON array in the content
            # Look for the first '[' and last ']'
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx+1]
            
            # Parse JSON
            data = json.loads(content)
            
            if not isinstance(data, list):
                logger.error(f"Expected list, got {type(data)}")
                return []
            
            entries = []
            for item in data:
                if isinstance(item, dict) and 'input' in item and 'expected_output' in item:
                    entries.append(DatasetEntry(
                        input=item['input'],
                        expected_output=item['expected_output'],
                        metadata=item.get('metadata', {})
                    ))
            
            return entries
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dataset response: {e}")
            logger.debug(f"Response content: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing dataset: {e}")
            return []
    
    def generate(self, num_samples: int, max_retries: int = 3) -> List[DatasetEntry]:
        """Generate dataset with specified number of samples.
        
        Uses chunked generation for large sample sizes to improve reliability.
        """
        logger.info(f"Generating {num_samples} dataset samples...")
        
        all_entries = []
        chunk_size = min(10, num_samples)  # Generate in chunks of 10 for reliability
        chunks_needed = (num_samples + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(chunks_needed):
            remaining = num_samples - len(all_entries)
            current_chunk_size = min(chunk_size, remaining)
            
            logger.info(f"Generating chunk {chunk_idx + 1}/{chunks_needed} ({current_chunk_size} samples)...")
            
            chunk_entries = self._generate_chunk(current_chunk_size, max_retries)
            
            if chunk_entries:
                all_entries.extend(chunk_entries)
                logger.info(f"Chunk {chunk_idx + 1} complete: {len(chunk_entries)} entries (total: {len(all_entries)})")
            else:
                logger.warning(f"Chunk {chunk_idx + 1} failed to generate entries")
            
            # Stop if we've generated enough
            if len(all_entries) >= num_samples:
                break
        
        if len(all_entries) >= num_samples * 0.5:  # Accept if we got at least 50%
            logger.info(f"Successfully generated {len(all_entries)} dataset entries")
            return all_entries[:num_samples]
        else:
            logger.error(f"Failed to generate sufficient dataset entries: {len(all_entries)}/{num_samples}")
            return []
    
    def _generate_chunk(self, chunk_size: int, max_retries: int) -> List[DatasetEntry]:
        """Generate a single chunk of dataset entries with robust fallback mechanisms."""
        
        errors = []
        
        # Try standard generation first
        for attempt in range(max_retries):
            prompt = self._build_generation_prompt(chunk_size)
            
            response = self.llm_client.query(
                prompt,
                system_message="You are a helpful assistant that generates high-quality test datasets. Always respond with valid JSON array format."
            )
            
            if not response.success:
                error_msg = f"Attempt {attempt + 1} failed: {response.error}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
            
            entries = self._parse_dataset_response(response.content)
            
            if len(entries) >= chunk_size * 0.5:  # Accept if we got at least 50% of chunk
                logger.info(f"Successfully generated {len(entries)} entries on attempt {attempt + 1}")
                return entries[:chunk_size]
            else:
                warning_msg = f"Only generated {len(entries)} entries in chunk (expected ~{chunk_size}), retrying..."
                logger.warning(warning_msg)
                errors.append(warning_msg)
        
        # Fallback 1: Try with simpler prompt
        logger.warning("Standard generation failed, trying simplified fallback prompt...")
        for attempt in range(2):
            simple_prompt = self._build_simple_fallback_prompt(chunk_size)
            
            response = self.llm_client.query(
                simple_prompt,
                system_message="Generate test cases in simple JSON format."
            )
            
            if response.success:
                entries = self._parse_dataset_response(response.content)
                if len(entries) >= chunk_size * 0.3:  # Lower threshold for fallback
                    logger.info(f"Fallback generation succeeded with {len(entries)} entries")
                    return entries[:chunk_size]
            else:
                errors.append(f"Fallback attempt {attempt + 1} failed: {response.error}")
        
        # Fallback 2: Try with even simpler prompt and lower expectations
        logger.warning("Simplified fallback failed, trying minimal prompt...")
        minimal_prompt = f"""Create {chunk_size} simple Q&A pairs for: {self.task_config.name}

Example format:
Q: [question]
A: [answer]

Create {chunk_size} examples:"""
        
        response = self.llm_client.query(minimal_prompt)
        if response.success and response.content:
            # Try to parse Q&A format
            entries = self._parse_qa_format(response.content)
            if len(entries) >= chunk_size * 0.2:
                logger.info(f"Minimal prompt generation succeeded with {len(entries)} entries")
                return entries[:chunk_size]
        
        # Final fallback: Generate minimal examples manually
        logger.error(f"All generation attempts failed. Errors: {errors}")
        logger.warning(f"Creating {chunk_size} minimal fallback entries to guarantee batch_size")
        return self._create_minimal_fallback_entries(chunk_size)
    
    def _parse_qa_format(self, content: str) -> List[DatasetEntry]:
        """Parse Q&A format response into dataset entries."""
        entries = []
        lines = content.strip().split('\n')
        
        current_input = None
        current_output = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for Q: or Question: prefix
            if line.upper().startswith('Q:') or line.upper().startswith('QUESTION:'):
                # Save previous entry if exists
                if current_input and current_output:
                    entries.append(DatasetEntry(
                        input=current_input,
                        expected_output=current_output,
                        metadata={"source": "qa_parsing"}
                    ))
                current_input = line.split(':', 1)[1].strip() if ':' in line else line
                current_output = None
            
            # Check for A: or Answer: prefix
            elif line.upper().startswith('A:') or line.upper().startswith('ANSWER:'):
                current_output = line.split(':', 1)[1].strip() if ':' in line else line
            
            # Handle numbered format (1., 2., etc.)
            elif line[0].isdigit() and '. ' in line[:5]:
                # This might be a numbered question
                if current_input and current_output:
                    entries.append(DatasetEntry(
                        input=current_input,
                        expected_output=current_output,
                        metadata={"source": "qa_parsing"}
                    ))
                parts = line.split('. ', 1)
                if len(parts) > 1:
                    current_input = parts[1]
                    current_output = None
        
        # Don't forget the last entry
        if current_input and current_output:
            entries.append(DatasetEntry(
                input=current_input,
                expected_output=current_output,
                metadata={"source": "qa_parsing"}
            ))
        
        return entries
    
    def _build_simple_fallback_prompt(self, num_samples: int) -> str:
        """Build a simpler fallback prompt for dataset generation."""
        return f"""Generate {num_samples} simple test cases for: {self.task_config.name}

Task: {self.task_config.description}

Return ONLY a JSON array like this:
[{{"input": "test input here", "expected_output": "expected answer here"}}]

Generate {num_samples} examples."""
    
    def _create_minimal_fallback_entries(self, num_samples: int) -> List[DatasetEntry]:
        """Create minimal fallback entries when LLM generation fails completely."""
        logger.warning(f"Creating {num_samples} minimal fallback entries for {self.task_config.name}")
        
        entries = []
        task_name = self.task_config.name.lower()
        
        # Create task-specific fallback examples
        if "reasoning" in task_name or "logic" in task_name:
            templates = [
                ("If A is taller than B, and B is taller than C, who is tallest?", "A is the tallest."),
                ("All cats are mammals. Is a cat a mammal?", "Yes, a cat is a mammal."),
                ("If it rains, the ground gets wet. It is raining. Is the ground wet?", "Yes, the ground is wet."),
                ("Three people: Alice, Bob, Charlie. Alice is older than Bob. Bob is older than Charlie. Who is youngest?", "Charlie is the youngest."),
                ("If all roses are flowers and some flowers are red, are all roses red?", "No, not all roses are necessarily red.")
            ]
        elif "sentiment" in task_name or "classification" in task_name:
            templates = [
                ("I love this product!", "positive"),
                ("This is terrible.", "negative"),
                ("It's okay, nothing special.", "neutral"),
                ("Best experience ever!", "positive"),
                ("I hate waiting.", "negative")
            ]
        else:
            # Generic templates
            templates = [
                (f"Example input 1 for {self.task_config.name}", "Expected output 1"),
                (f"Example input 2 for {self.task_config.name}", "Expected output 2"),
                (f"Example input 3 for {self.task_config.name}", "Expected output 3"),
                (f"Example input 4 for {self.task_config.name}", "Expected output 4"),
                (f"Example input 5 for {self.task_config.name}", "Expected output 5")
            ]
        
        # Generate entries from templates
        for i in range(num_samples):
            template = templates[i % len(templates)]
            # Add variation to make them unique
            variation = f" (variant {i//len(templates) + 1})" if i >= len(templates) else ""
            entries.append(DatasetEntry(
                input=template[0] + variation,
                expected_output=template[1],
                metadata={"source": "fallback_generation", "template_index": i % len(templates)}
            ))
        
        return entries
    
    def validate_dataset(self, entries: List[DatasetEntry]) -> Tuple[bool, str]:
        """Validate generated dataset for quality."""
        if not entries:
            return False, "Dataset is empty"
        
        # Check for duplicates
        inputs_seen = set()
        duplicates = 0
        for entry in entries:
            if entry.input in inputs_seen:
                duplicates += 1
            inputs_seen.add(entry.input)
        
        if duplicates > len(entries) * 0.1:  # More than 10% duplicates
            return False, f"Too many duplicates: {duplicates}/{len(entries)}"
        
        # Check for empty entries
        empty_count = sum(1 for e in entries if not e.input or not e.expected_output)
        if empty_count > 0:
            return False, f"Found {empty_count} empty entries"
        
        return True, f"Dataset valid: {len(entries)} entries, {duplicates} duplicates"
    
    def save_dataset(self, entries: List[DatasetEntry], filepath: str):
        """Save dataset to JSON file."""
        data = [entry.to_dict() for entry in entries]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[DatasetEntry]:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        entries = [DatasetEntry.from_dict(item) for item in data]
        logger.info(f"Loaded {len(entries)} entries from {filepath}")
        return entries


if __name__ == "__main__":
    # Test dataset generation
    from config_manager import Config, LLMConfig, TaskConfig
    
    config = Config(
        optimizer_llm=LLMConfig(model="qwen/qwen3-8b:free"),
        target_llm=LLMConfig(model="qwen/qwen3-8b:free"),
        experiment=None,
        task=TaskConfig(
            name="sentiment_analysis",
            description="Classify text sentiment as positive or negative"
        ),
        metric=None,
        context=None,
        storage=None
    )
    
    client = LLMClient(config.optimizer_llm)
    generator = DatasetGenerator(client, config.task)
    
    print("Testing dataset generation...")
    entries = generator.generate(5)
    
    print(f"\nGenerated {len(entries)} entries:")
    for i, entry in enumerate(entries[:3], 1):
        print(f"\n{i}. Input: {entry.input[:100]}...")
        print(f"   Expected: {entry.expected_output}")

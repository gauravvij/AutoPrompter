"""
Configuration Management Module for Autonomous Prompt Optimization System.
Handles loading, validation, and dynamic overrides of configuration parameters.
"""

import yaml
import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


@dataclass
class LLMConfig:
    """Configuration for LLM models."""
    model: str
    api_base: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = self._load_api_key()
    
    def _load_api_key(self) -> str:
        """Load API key from config file or environment."""
        config_path = "/root/.config/openrouter/config"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('api_key', '')
            except (json.JSONDecodeError, IOError):
                pass
        
        # Fallback to environment variable
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        if api_key:
            return api_key
        
        raise ValueError(
            "OpenRouter API key not found. Please set it in /root/.config/openrouter/config "
            "or as OPENROUTER_API_KEY environment variable."
        )


@dataclass
class ExperimentConfig:
    """Configuration for experiment parameters."""
    max_iterations: int = 100
    convergence_threshold: float = 0.95
    min_improvement: float = 0.01
    batch_size: int = 5


@dataclass
class TaskConfig:
    """Configuration for the optimization task."""
    name: str = "text_classification"
    description: str = "Classify text into positive or negative sentiment"
    initial_prompt: str = "Analyze the sentiment of the following text and respond with only 'positive' or 'negative'."


@dataclass
class MetricConfig:
    """Configuration for evaluation metrics."""
    type: str = "accuracy"  # accuracy, f1, exact_match, contains, semantic_similarity
    target_score: float = 0.95


@dataclass
class ContextConfig:
    """Configuration for context management."""
    max_experiments_in_context: int = 20
    compression_threshold: int = 50


@dataclass
class StorageConfig:
    """Configuration for storage paths."""
    ledger_file: str = "experiment_ledger.json"
    dataset_file: str = "generated_dataset.json"
    results_dir: str = "results"
    checkpoint_interval: int = 10


@dataclass
class Config:
    """Main configuration container."""
    optimizer_llm: LLMConfig
    target_llm: LLMConfig
    experiment: ExperimentConfig
    task: TaskConfig
    metric: MetricConfig
    context: ContextConfig
    storage: StorageConfig
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            optimizer_llm=LLMConfig(**data.get('optimizer_llm', {})),
            target_llm=LLMConfig(**data.get('target_llm', {})),
            experiment=ExperimentConfig(**data.get('experiment', {})),
            task=TaskConfig(**data.get('task', {})),
            metric=MetricConfig(**data.get('metric', {})),
            context=ContextConfig(**data.get('context', {})),
            storage=StorageConfig(**data.get('storage', {}))
        )
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        data = {
            'optimizer_llm': asdict(self.optimizer_llm),
            'target_llm': asdict(self.target_llm),
            'experiment': asdict(self.experiment),
            'task': asdict(self.task),
            'metric': asdict(self.metric),
            'context': asdict(self.context),
            'storage': asdict(self.storage)
        }
        # Remove API keys from saved config for security
        data['optimizer_llm']['api_key'] = None
        data['target_llm']['api_key'] = None
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def override_from_dict(self, overrides: Dict[str, Any]):
        """Apply dynamic overrides to configuration."""
        for key, value in overrides.items():
            if '.' in key:
                # Nested override (e.g., 'experiment.max_iterations')
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                # Top-level override
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate LLM configs
        if not self.optimizer_llm.model:
            errors.append("optimizer_llm.model is required")
        if not self.target_llm.model:
            errors.append("target_llm.model is required")
        if not self.optimizer_llm.api_key:
            errors.append("optimizer_llm.api_key is required")
        if not self.target_llm.api_key:
            errors.append("target_llm.api_key is required")
        
        # Validate experiment config
        if self.experiment.max_iterations < 1:
            errors.append("experiment.max_iterations must be >= 1")
        if self.experiment.batch_size < 1:
            errors.append("experiment.batch_size must be >= 1")
        
        # Validate task config
        if not self.task.name:
            errors.append("task.name is required")
        if not self.task.initial_prompt:
            errors.append("task.initial_prompt is required")
        
        # Validate metric config - now supports 'auto' for optimizer-defined metrics
        valid_metrics = ['accuracy', 'f1', 'exact_match', 'contains', 'semantic_similarity', 'auto']
        if self.metric.type not in valid_metrics:
            errors.append(f"metric.type must be one of {valid_metrics}")
        
        return errors


def load_config(filepath: str = "config.yaml", overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load and validate configuration with optional overrides."""
    config = Config.from_yaml(filepath)
    
    if overrides:
        config.override_from_dict(overrides)
    
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    return config


if __name__ == "__main__":
    # Test configuration loading
    import sys
    
    # Create test config
    config = Config(
        optimizer_llm=LLMConfig(
            model="qwen/qwen3.5-9b",
            api_key="test-key"
        ),
        target_llm=LLMConfig(
            model="qwen/qwen3.5-9b",
            api_key="test-key"
        ),
        experiment=ExperimentConfig(max_iterations=10),
        task=TaskConfig(
            name="test_task",
            initial_prompt="Test prompt"
        ),
        metric=MetricConfig(type="accuracy"),
        context=ContextConfig(),
        storage=StorageConfig()
    )
    
    # Test validation
    errors = config.validate()
    if errors:
        print("Validation errors:", errors)
        sys.exit(1)
    
    print("Configuration validation passed!")
    print(f"Optimizer LLM: {config.optimizer_llm.model}")
    print(f"Target LLM: {config.target_llm.model}")
    print(f"Max iterations: {config.experiment.max_iterations}")

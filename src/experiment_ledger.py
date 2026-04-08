"""
Experiment Ledger Module for persistent storage of experiment records.
Prevents duplicate experiments and maintains history.
Includes semantic duplicate detection using embeddings.
"""

import json
import os
import hashlib
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import sentence-transformers for semantic duplicate detection
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_DETECTION_AVAILABLE = True
except ImportError:
    SEMANTIC_DETECTION_AVAILABLE = False
    logger.warning("sentence-transformers not available, semantic duplicate detection disabled")


@dataclass
class ExperimentRecord:
    """Single experiment record."""
    iteration: int
    prompt: str
    inputs: List[str]
    expected_outputs: List[str]
    actual_outputs: List[str]
    metric_scores: List[float]
    mean_score: float
    metric_type: str = "accuracy"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    experiment_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.experiment_hash:
            self.experiment_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute unique hash for this experiment to detect duplicates."""
        content = f"{self.prompt}|{','.join(self.inputs)}|{','.join(self.expected_outputs)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRecord':
        return cls(**data)


class ExperimentLedger:
    """Persistent storage for experiment records with duplicate detection.
    
    Includes both exact hash-based deduplication and semantic similarity detection
    using sentence embeddings to catch near-duplicate prompts.
    """
    
    # Class-level embedding model (shared across instances)
    _embedding_model = None
    _embedding_model_name = "all-MiniLM-L6-v2"  # Lightweight, fast model
    
    def __init__(self, storage_config, semantic_similarity_threshold: float = 0.95):
        """Initialize ledger with storage configuration.
        
        Args:
            storage_config: Storage configuration
            semantic_similarity_threshold: Cosine similarity threshold for semantic duplicates (0.0-1.0)
        """
        self.storage_config = storage_config
        self.ledger_file = storage_config.ledger_file
        self.checkpoint_interval = storage_config.checkpoint_interval
        self.records: List[ExperimentRecord] = []
        self.seen_hashes: set = set()
        
        # Semantic duplicate detection settings
        self.semantic_threshold = semantic_similarity_threshold
        self.prompt_embeddings: Dict[str, List[float]] = {}  # prompt_hash -> embedding
        self._init_embedding_model()
        
        self._load_ledger()
    
    def _init_embedding_model(self):
        """Initialize the embedding model for semantic duplicate detection."""
        if not SEMANTIC_DETECTION_AVAILABLE:
            return
        
        if ExperimentLedger._embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {self._embedding_model_name}")
                ExperimentLedger._embedding_model = SentenceTransformer(self._embedding_model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                ExperimentLedger._embedding_model = None
    
    def _compute_embedding(self, prompt: str) -> Optional[List[float]]:
        """Compute embedding for a prompt.
        
        Args:
            prompt: The prompt text to embed
        
        Returns:
            Embedding vector as list of floats, or None if model unavailable
        """
        if not SEMANTIC_DETECTION_AVAILABLE or ExperimentLedger._embedding_model is None:
            return None
        
        try:
            embedding = ExperimentLedger._embedding_model.encode(prompt, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.debug(f"Failed to compute embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity in range [-1, 1]
        """
        if not SEMANTIC_DETECTION_AVAILABLE:
            return 0.0
        
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(v1, v2) / (norm1 * norm2))
        except Exception as e:
            logger.debug(f"Failed to compute cosine similarity: {e}")
            return 0.0
    
    def is_semantic_duplicate(self, prompt: str, threshold: Optional[float] = None) -> bool:
        """Check if a prompt is semantically similar to any existing prompt.
        
        Args:
            prompt: The prompt to check
            threshold: Optional override for similarity threshold
        
        Returns:
            True if a semantic duplicate is found
        """
        if not SEMANTIC_DETECTION_AVAILABLE or ExperimentLedger._embedding_model is None:
            return False  # Fall back to hash-only detection
        
        threshold = threshold or self.semantic_threshold
        
        # Compute embedding for new prompt
        new_embedding = self._compute_embedding(prompt)
        if new_embedding is None:
            return False
        
        # Compare against all existing prompt embeddings
        for existing_hash, existing_embedding in self.prompt_embeddings.items():
            similarity = self._cosine_similarity(new_embedding, existing_embedding)
            if similarity >= threshold:
                logger.info(f"Semantic duplicate detected (similarity: {similarity:.3f}) - "
                           f"matches hash {existing_hash[:8]}...")
                return True
        
        return False
    
    def add_prompt_embedding(self, prompt: str, prompt_hash: str):
        """Compute and store embedding for a prompt.
        
        Args:
            prompt: The prompt text
            prompt_hash: The hash identifier for the prompt
        """
        if not SEMANTIC_DETECTION_AVAILABLE or ExperimentLedger._embedding_model is None:
            return
        
        embedding = self._compute_embedding(prompt)
        if embedding is not None:
            self.prompt_embeddings[prompt_hash] = embedding
            logger.debug(f"Stored embedding for prompt hash {prompt_hash[:8]}...")
    
    def _load_ledger(self):
        """Load existing ledger from file."""
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, 'r') as f:
                    data = json.load(f)
                
                self.records = [ExperimentRecord.from_dict(r) for r in data.get('records', [])]
                self.seen_hashes = {r.experiment_hash for r in self.records}
                
                logger.info(f"Loaded {len(self.records)} records from ledger")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load ledger: {e}")
                self.records = []
                self.seen_hashes = set()
    
    def _save_ledger(self):
        """Save ledger to file."""
        try:
            data = {
                'records': [r.to_dict() for r in self.records],
                'metadata': {
                    'total_records': len(self.records),
                    'last_saved': datetime.now().isoformat()
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.ledger_file) if os.path.dirname(self.ledger_file) else '.', exist_ok=True)
            
            with open(self.ledger_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.records)} records to ledger")
        except IOError as e:
            logger.error(f"Failed to save ledger: {e}")
    
    def is_duplicate(self, record: ExperimentRecord) -> bool:
        """Check if experiment is a duplicate (exact or semantic).
        
        First checks exact hash match, then falls back to semantic similarity.
        """
        # Check exact hash match first
        if record.experiment_hash in self.seen_hashes:
            logger.debug(f"Exact duplicate detected: {record.experiment_hash}")
            return True
        
        # Check semantic similarity
        if self.is_semantic_duplicate(record.prompt):
            logger.info(f"Semantic duplicate detected for prompt (hash: {record.experiment_hash[:8]}...)")
            return True
        
        return False
    
    def is_duplicate_experiment(self, record: ExperimentRecord) -> bool:
        """Alias for is_duplicate for compatibility."""
        return self.is_duplicate(record)
    
    def add_experiment(self, record: ExperimentRecord) -> bool:
        """Alias for add_record for compatibility."""
        return self.add_record(record)
    
    def add_record(self, record: ExperimentRecord) -> bool:
        """Add record to ledger if not duplicate. Returns True if added."""
        if self.is_duplicate(record):
            logger.debug(f"Duplicate experiment detected: {record.experiment_hash}")
            return False
        
        self.records.append(record)
        self.seen_hashes.add(record.experiment_hash)
        
        # Store embedding for semantic duplicate detection
        self.add_prompt_embedding(record.prompt, record.experiment_hash)
        
        # Auto-save on checkpoint interval
        if len(self.records) % self.checkpoint_interval == 0:
            self._save_ledger()
        
        return True
    
    def get_records(self, iteration: Optional[int] = None) -> List[ExperimentRecord]:
        """Get records, optionally filtered by iteration."""
        if iteration is not None:
            return [r for r in self.records if r.iteration == iteration]
        return self.records
    
    def get_all_experiments(self) -> List[ExperimentRecord]:
        """Get all experiment records."""
        return self.records
    
    def get_all_records(self) -> List[ExperimentRecord]:
        """Alias for get_all_experiments."""
        return self.get_all_experiments()
    
    def get_best_record(self) -> Optional[ExperimentRecord]:
        """Get record with highest metric score."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.metric_score)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        if not self.records:
            return {
                'total_records': 0,
                'unique_prompts': 0,
                'best_score': 0.0,
                'average_score': 0.0
            }
        
        scores = [r.metric_score for r in self.records]
        unique_prompts = len(set(r.prompt for r in self.records))
        
        return {
            'total_records': len(self.records),
            'unique_prompts': unique_prompts,
            'best_score': max(scores),
            'average_score': sum(scores) / len(scores),
            'iterations': max(r.iteration for r in self.records)
        }
    
    def close(self):
        """Save ledger and cleanup."""
        self._save_ledger()


if __name__ == "__main__":
    # Test ledger functionality
    from config_manager import StorageConfig
    
    storage = StorageConfig(ledger_file="test_ledger.json")
    ledger = ExperimentLedger(storage)
    
    # Add test records
    for i in range(5):
        record = ExperimentRecord(
            iteration=1,
            prompt="Test prompt",
            test_input=f"Input {i}",
            expected_output="positive",
            actual_output="positive" if i % 2 == 0 else "negative",
            metric_score=1.0 if i % 2 == 0 else 0.0,
            metric_type="accuracy"
        )
        ledger.add_record(record)
    
    # Test statistics
    stats = ledger.get_statistics()
    print(f"Ledger statistics: {stats}")
    
    # Test duplicate detection
    duplicate = ExperimentRecord(
        iteration=2,
        prompt="Test prompt",
        test_input="Input 0",
        expected_output="positive",
        actual_output="positive",
        metric_score=1.0,
        metric_type="accuracy"
    )
    is_dup = ledger.is_duplicate(duplicate)
    print(f"Duplicate detected: {is_dup}")
    
    ledger.close()
    print("Ledger test completed!")
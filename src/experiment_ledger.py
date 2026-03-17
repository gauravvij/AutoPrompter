"""
Experiment Ledger Module for persistent storage of experiment records.
Prevents duplicate experiments and maintains history.
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
    """Persistent storage for experiment records with duplicate detection."""
    
    def __init__(self, storage_config):
        """Initialize ledger with storage configuration."""
        self.storage_config = storage_config
        self.ledger_file = storage_config.ledger_file
        self.checkpoint_interval = storage_config.checkpoint_interval
        self.records: List[ExperimentRecord] = []
        self.seen_hashes: set = set()
        self._load_ledger()
    
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
        """Check if experiment is a duplicate."""
        return record.experiment_hash in self.seen_hashes
    
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
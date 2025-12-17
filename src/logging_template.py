"""
Template for structured logging and results saving (CSV/JSON per subject + seeds)
"""

import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml


class ResultsLogger:
    """Template for logging and saving results per subject with seed tracking"""
    
    def __init__(self, config_path: str, output_dir: str = "results"):
        """
        Initialize results logger
        
        Args:
            config_path: Path to YAML config file
            output_dir: Directory to save results
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_subject_results(self, subject_id: str, results: Dict[str, Any], seed: int):
        """
        Save results for a single subject
        
        Args:
            subject_id: Subject identifier
            results: Dictionary with results
            seed: Random seed used
        """
        # Create subject-specific directory
        subject_dir = self.output_dir / f"subject_{subject_id}"
        subject_dir.mkdir(exist_ok=True)
        
        # Add metadata
        results_with_meta = {
            "metadata": {
                "subject_id": subject_id,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "config": self.config
            },
            "results": results
        }
        
        # Save as JSON
        json_file = subject_dir / f"seed_{seed}_results.json"
        with open(json_file, 'w') as f:
            json.dump(results_with_meta, f, indent=2)
        self.logger.info(f"Saved JSON results for subject {subject_id}, seed {seed}")
        
        # Save as CSV (flattened)
        self._save_as_csv(subject_dir, subject_id, seed, results)
    
    @staticmethod
    def _save_as_csv(subject_dir: Path, subject_id: str, seed: int, results: Dict):
        """Save flattened results as CSV"""
        csv_file = subject_dir / "results.csv"
        
        # Flatten nested dictionary
        flat_results = ResultsLogger._flatten_dict(results)
        flat_results['subject_id'] = subject_id
        flat_results['seed'] = seed
        flat_results['timestamp'] = datetime.now().isoformat()
        
        # Write CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_results.keys())
            
            # Write header only if file is empty
            if csv_file.stat().st_size == 0:
                writer.writeheader()
            
            writer.writerow(flat_results)
    
    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ResultsLogger._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_seeds_summary(self, seeds_used: List[int]):
        """Log summary of seeds used in experiment"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "seeds": seeds_used,
            "num_seeds": len(seeds_used),
            "seed_config": self.config.get('seeds', {})
        }
        
        seeds_file = self.output_dir / "seeds_summary.json"
        with open(seeds_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Seeds summary: {len(seeds_used)} seeds used")
    
    def log_experiment_config(self):
        """Save experiment configuration"""
        config_file = self.output_dir / "experiment_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f)
        self.logger.info("Saved experiment configuration")


# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = ResultsLogger(
        config_path="configs/paper1.yaml",
        output_dir="results"
    )
    
    # Log experiment config
    logger.log_experiment_config()
    
    # Example: Save results for multiple subjects and seeds
    seeds_list = [42, 123, 456]
    subjects = ["S01", "S02", "S03"]
    
    for subject in subjects:
        for seed in seeds_list:
            # Example results structure
            results = {
                "accuracy": 0.85,
                "precision": 0.87,
                "recall": 0.83,
                "f1": 0.85,
                "confusion_matrix": [[45, 5], [8, 42]],
                "training_time": 120.5
            }
            
            logger.save_subject_results(subject, results, seed)
    
    # Log seeds summary
    logger.log_seeds_summary(seeds_list)

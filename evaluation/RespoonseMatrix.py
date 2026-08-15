"""
Response Matrix - The Universal Data Structure for AI Evaluation

Rows: AI systems (models, agents, or variants)
Columns: Evaluation items (questions, tasks, prompts)
Cells: Performance scores, preferences, or binary outcomes
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResponseMatrix:
    """
    Universal response matrix for AI evaluation.
    
    Attributes:
        systems: List of system names (rows)
        items: List of item IDs or questions (columns)
        matrix: 2D numpy array of responses
        metadata: Additional metadata about each cell
        response_type: 'binary', 'continuous', 'ordinal', 'preference'
    """
    systems: List[str]
    items: List[str]
    matrix: np.ndarray
    metadata: Dict[Tuple[int, int], Dict] = field(default_factory=dict)
    response_type: str = "continuous"
    
    def __post_init__(self):
        """Validate matrix dimensions."""
        if len(self.systems) != self.matrix.shape[0]:
            raise ValueError("Number of systems must match matrix rows")
        if len(self.items) != self.matrix.shape[1]:
            raise ValueError("Number of items must match matrix columns")
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResponseMatrix':
        """Create ResponseMatrix from dictionary."""
        return cls(
            systems=data.get("systems", []),
            items=data.get("items", []),
            matrix=np.array(data.get("matrix", [])),
            metadata=data.get("metadata", {}),
            response_type=data.get("response_type", "continuous")
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "systems": self.systems,
            "items": self.items,
            "matrix": self.matrix.tolist(),
            "metadata": self.metadata,
            "response_type": self.response_type,
            "shape": self.shape
        }
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix shape (n_systems, n_items)."""
        return self.matrix.shape
    
    @property
    def n_systems(self) -> int:
        """Number of systems evaluated."""
        return len(self.systems)
    
    @property
    def n_items(self) -> int:
        """Number of evaluation items."""
        return len(self.items)
    
    @property
    def density(self) -> float:
        """Proportion of non-missing responses."""
        non_missing = np.sum(~np.isnan(self.matrix))
        return non_missing / (self.n_systems * self.n_items)
    
    def get_system_scores(self, system_idx: int) -> np.ndarray:
        """Get all scores for a specific system."""
        return self.matrix[system_idx, :]
    
    def get_item_scores(self, item_idx: int) -> np.ndarray:
        """Get all scores for a specific item."""
        return self.matrix[:, item_idx]
    
    def get_system_mean(self, system_idx: int) -> float:
        """Get mean score for a system (ignoring NaN)."""
        scores = self.get_system_scores(system_idx)
        return np.nanmean(scores)
    
    def get_item_mean(self, item_idx: int) -> float:
        """Get mean score for an item (ignoring NaN)."""
        scores = self.get_item_scores(item_idx)
        return np.nanmean(scores)
    
    def compute_item_difficulty(self) -> np.ndarray:
        """Compute item difficulty (lower = easier)."""
        return 1.0 - np.nanmean(self.matrix, axis=0)
    
    def compute_system_ability(self) -> np.ndarray:
        """Compute system ability (higher = better)."""
        return np.nanmean(self.matrix, axis=1)
    
    def summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "n_systems": self.n_systems,
            "n_items": self.n_items,
            "density": self.density,
            "mean_score": np.nanmean(self.matrix),
            "std_score": np.nanstd(self.matrix),
            "min_score": np.nanmin(self.matrix),
            "max_score": np.nanmax(self.matrix),
            "system_means": self.compute_system_ability().tolist(),
            "item_difficulties": self.compute_item_difficulty().tolist()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame(
            self.matrix,
            index=self.systems,
            columns=self.items
        )
        return df
    
    @classmethod
    def from_evaluation_results(cls, results: List[Dict]) -> 'ResponseMatrix':
        """
        Create ResponseMatrix from evaluation results.
        
        Args:
            results: List of evaluation results with keys:
                - system: System name
                - item: Item ID
                - score: Response score
                - metadata: Optional metadata
        """
        systems = list(set(r["system"] for r in results))
        items = list(set(r["item"] for r in results))
        systems.sort()
        items.sort()
        
        # Create matrix with NaN
        matrix = np.full((len(systems), len(items)), np.nan)
        metadata = {}
        
        system_to_idx = {s: i for i, s in enumerate(systems)}
        item_to_idx = {it: j for j, it in enumerate(items)}
        
        for r in results:
            i = system_to_idx[r["system"]]
            j = item_to_idx[r["item"]]
            matrix[i, j] = r["score"]
            metadata[(i, j)] = r.get("metadata", {})
        
        return cls(
            systems=systems,
            items=items,
            matrix=matrix,
            metadata=metadata,
            response_type="continuous"
        )


class ResponseMatrixBuilder:
    """Builder for creating Response Matrices incrementally."""
    
    def __init__(self):
        self.systems = []
        self.items = []
        self.data = defaultdict(dict)
        self.metadata = defaultdict(dict)
    
    def add_result(self, system: str, item: str, score: float, metadata: Dict = None):
        """Add a single evaluation result."""
        if system not in self.systems:
            self.systems.append(system)
        if item not in self.items:
            self.items.append(item)
        
        self.data[system][item] = score
        if metadata:
            self.metadata[(system, item)] = metadata
    
    def build(self) -> ResponseMatrix:
        """Build the ResponseMatrix."""
        systems = sorted(self.systems)
        items = sorted(self.items)
        
        matrix = np.full((len(systems), len(items)), np.nan)
        metadata = {}
        
        for i, system in enumerate(systems):
            for j, item in enumerate(items):
                if item in self.data[system]:
                    matrix[i, j] = self.data[system][item]
                    if (system, item) in self.metadata:
                        metadata[(i, j)] = self.metadata[(system, item)]
        
        return ResponseMatrix(
            systems=systems,
            items=items,
            matrix=matrix,
            metadata=metadata
        )


class ResponseMatrixAnalyzer:
    """Analyze Response Matrix for insights."""
    
    def __init__(self, matrix: ResponseMatrix):
        self.matrix = matrix
    
    def compute_difficulty_distribution(self) -> Dict:
        """Analyze distribution of item difficulties."""
        difficulties = self.matrix.compute_item_difficulty()
        return {
            "mean": np.nanmean(difficulties),
            "std": np.nanstd(difficulties),
            "min": np.nanmin(difficulties),
            "max": np.nanmax(difficulties),
            "histogram": np.histogram(difficulties, bins=10)
        }
    
    def compute_ability_distribution(self) -> Dict:
        """Analyze distribution of system abilities."""
        abilities = self.matrix.compute_system_ability()
        return {
            "mean": np.nanmean(abilities),
            "std": np.nanstd(abilities),
            "min": np.nanmin(abilities),
            "max": np.nanmax(abilities)
        }
    
    def identify_anomalous_items(self, threshold: float = 2.0) -> List[Dict]:
        """
        Identify items with unusual difficulty patterns.
        
        Args:
            threshold: Z-score threshold for anomalies.
        
        Returns:
            List of anomalous items with their scores.
        """
        difficulties = self.matrix.compute_item_difficulty()
        mean = np.nanmean(difficulties)
        std = np.nanstd(difficulties)
        
        anomalies = []
        for i, diff in enumerate(difficulties):
            z_score = (diff - mean) / std if std > 0 else 0
            if abs(z_score) > threshold:
                anomalies.append({
                    "item": self.matrix.items[i],
                    "difficulty": diff,
                    "z_score": z_score,
                    "mean_score": self.matrix.get_item_mean(i)
                })
        
        return anomalies
    
    def compute_bias_analysis(self, group_col: str) -> Dict:
        """
        Analyze bias across groups using Differential Item Functioning (DIF).
        
        Args:
            group_col: Column name in metadata for grouping.
        
        Returns:
            DIF statistics for each item.
        """
        # This is a simplified version - DIF analysis is complex
        results = {}
        
        for j, item in enumerate(self.matrix.items):
            scores_by_group = {}
            for i, system in enumerate(self.matrix.systems):
                group = self.matrix.metadata.get((i, j), {}).get(group_col, "unknown")
                if group not in scores_by_group:
                    scores_by_group[group] = []
                if not np.isnan(self.matrix.matrix[i, j]):
                    scores_by_group[group].append(self.matrix.matrix[i, j])
            
            # Compute DIF statistic (simplified)
            if len(scores_by_group) >= 2:
                means = {g: np.mean(s) for g, s in scores_by_group.items() if s}
                if means:
                    results[item] = {
                        "groups": means,
                        "max_diff": max(means.values()) - min(means.values())
                    }
        
        return results
    
    def save_analysis(self, output_dir: str = "analysis_results"):
        """Save analysis results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        with open(f"{output_dir}/matrix_summary.json", "w") as f:
            json.dump(self.matrix.summary(), f, indent=2)
        
        # Save difficulties
        difficulties = {
            "items": self.matrix.items,
            "difficulties": self.matrix.compute_item_difficulty().tolist()
        }
        with open(f"{output_dir}/item_difficulties.json", "w") as f:
            json.dump(difficulties, f, indent=2)
        
        # Save abilities
        abilities = {
            "systems": self.matrix.systems,
            "abilities": self.matrix.compute_system_ability().tolist()
        }
        with open(f"{output_dir}/system_abilities.json", "w") as f:
            json.dump(abilities, f, indent=2)
        
        # Save matrix as CSV
        df = self.matrix.to_dataframe()
        df.to_csv(f"{output_dir}/response_matrix.csv")
        
        logger.info(f" Analysis saved to {output_dir}/")
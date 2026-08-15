"""
Construct Validity Analysis for AI Evaluation

Measures:
- Convergent validity
- Discriminant validity
- Construct representation
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidityAnalyzer:
    """
    Analyze construct validity of evaluation measurements.
    """
    
    def __init__(self, response_matrix):
        self.matrix = response_matrix
    
    def compute_convergent_validity(self, external_measure: np.ndarray) -> Dict:
        """
        Compute convergent validity with external measure.
        
        Args:
            external_measure: External validation measure
        
        Returns:
            Correlation statistics
        """
        system_scores = self.matrix.compute_system_ability()
        
        # Remove NaN values
        valid_idx = ~np.isnan(system_scores) & ~np.isnan(external_measure)
        scores = system_scores[valid_idx]
        external = external_measure[valid_idx]
        
        if len(scores) < 2:
            return {"error": "Insufficient data for correlation"}
        
        pearson_corr, pearson_p = pearsonr(scores, external)
        spearman_corr, spearman_p = spearmanr(scores, external)
        
        return {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "n_valid": len(scores)
        }
    
    def compute_discriminant_validity(self, other_measure: np.ndarray) -> Dict:
        """
        Compute discriminant validity with different construct.
        
        Should be lower than convergent validity.
        """
        system_scores = self.matrix.compute_system_ability()
        
        valid_idx = ~np.isnan(system_scores) & ~np.isnan(other_measure)
        scores = system_scores[valid_idx]
        other = other_measure[valid_idx]
        
        if len(scores) < 2:
            return {"error": "Insufficient data"}
        
        pearson_corr, pearson_p = pearsonr(scores, other)
        
        return {
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p,
            "n_valid": len(scores)
        }
    
    def analyze_construct_representation(self) -> Dict:
        """
        Analyze how well items represent the construct.
        
        Returns:
            - Item-construct correlations
            - Items that may not align with the construct
        """
        system_scores = self.matrix.compute_system_ability()
        n_items = self.matrix.n_items
        
        item_correlations = []
        for j in range(n_items):
            item_scores = self.matrix.matrix[:, j]
            valid = ~np.isnan(item_scores) & ~np.isnan(system_scores)
            
            if np.sum(valid) > 1:
                corr, _ = pearsonr(item_scores[valid], system_scores[valid])
                item_correlations.append({
                    "item": self.matrix.items[j],
                    "correlation": corr if not np.isnan(corr) else 0
                })
        
        # Identify poor items (correlation < 0.3)
        poor_items = [item for item in item_correlations if item["correlation"] < 0.3]
        
        return {
            "item_correlations": item_correlations,
            "mean_correlation": np.mean([i["correlation"] for i in item_correlations]),
            "poor_items": poor_items
        }
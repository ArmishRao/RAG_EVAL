"""
Reliability Analysis for AI Evaluation

Measures:
- Test-retest reliability
- Inter-rater reliability
- Internal consistency (Cronbach's alpha)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReliabilityAnalyzer:
    """
    Analyze reliability of evaluation measurements.
    """
    
    def __init__(self, response_matrix):
        self.matrix = response_matrix
    
    def compute_cronbach_alpha(self) -> float:
        """
        Compute Cronbach's alpha for internal consistency.
        
        α = (k/(k-1)) * (1 - Σ(σ_i²)/σ_total²)
        """
        n_items = self.matrix.n_items
        n_systems = self.matrix.n_systems
        
        if n_items < 2:
            return np.nan
        
        # Compute item variances
        item_variances = []
        for j in range(n_items):
            item_scores = self.matrix.matrix[:, j]
            valid = ~np.isnan(item_scores)
            if np.sum(valid) > 1:
                item_variances.append(np.var(item_scores[valid]))
        
        if len(item_variances) < 2:
            return np.nan
        
        # Compute total variance
        system_scores = self.matrix.compute_system_ability()
        valid = ~np.isnan(system_scores)
        total_variance = np.var(system_scores[valid])
        
        # Cronbach's alpha
        sum_item_variances = sum(item_variances)
        k = len(item_variances)
        
        if total_variance == 0:
            return np.nan
        
        alpha = (k / (k - 1)) * (1 - (sum_item_variances / total_variance))
        
        return alpha
    
    def compute_split_half_reliability(self) -> Dict:
        """
        Compute split-half reliability.
        
        Splits items into two halves and correlates scores.
        """
        n_items = self.matrix.n_items
        
        if n_items < 4:
            return {"error": "Too few items for split-half reliability"}
        
        # Split into odd and even items
        odd_items = list(range(0, n_items, 2))
        even_items = list(range(1, n_items, 2))
        
        # Compute scores for each half
        odd_scores = np.nanmean(self.matrix.matrix[:, odd_items], axis=1)
        even_scores = np.nanmean(self.matrix.matrix[:, even_items], axis=1)
        
        # Remove NaN values
        valid = ~np.isnan(odd_scores) & ~np.isnan(even_scores)
        
        if np.sum(valid) < 2:
            return {"error": "Insufficient valid data"}
        
        correlation, p_value = pearsonr(odd_scores[valid], even_scores[valid])
        
        # Spearman-Brown correction
        spearman_brown = (2 * correlation) / (1 + correlation)
        
        return {
            "correlation": correlation,
            "p_value": p_value,
            "spearman_brown": spearman_brown,
            "n_valid": np.sum(valid)
        }
    
    def compute_measurement_error(self) -> Dict:
        """
        Compute measurement error statistics.
        
        Returns:
            - Standard Error of Measurement (SEM)
            - Reliability coefficient
        """
        # Cronbach's alpha as reliability estimate
        alpha = self.compute_cronbach_alpha()
        
        if np.isnan(alpha):
            return {"error": "Could not compute reliability"}
        
        # Standard deviation of scores
        scores = self.matrix.compute_system_ability()
        valid = ~np.isnan(scores)
        sd = np.std(scores[valid])
        
        # SEM = SD * sqrt(1 - reliability)
        sem = sd * np.sqrt(1 - alpha)
        
        return {
            "reliability": alpha,
            "standard_deviation": sd,
            "standard_error_measurement": sem,
            "interpretation": f"Measurement error: ±{sem*1.96:.2f} (95% CI)"
        }
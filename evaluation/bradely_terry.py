"""
Bradley-Terry Models for Preference-Based Evaluation

Used for:
- Chatbot Arena style comparisons
- Pairwise preference data
- Elo rating systems
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BradleyTerryModel:
    """
    Bradley-Terry Model for pairwise comparisons.
    
    P(i beats j) = exp(θ_i) / (exp(θ_i) + exp(θ_j))
    """
    
    def __init__(self):
        self.parameters = None
        self.systems = []
        self.fitted = False
    
    def fit(self, comparisons: List[Tuple[str, str, int]]):
        """
        Fit Bradley-Terry model from pairwise comparisons.
        
        Args:
            comparisons: List of (winner, loser, count) tuples
        """
        logger.info("Fitting Bradley-Terry Model...")
        
        # Extract unique systems
        systems = set()
        for w, l, _ in comparisons:
            systems.add(w)
            systems.add(l)
        self.systems = sorted(list(systems))
        n = len(self.systems)
        
        system_to_idx = {s: i for i, s in enumerate(self.systems)}
        
        # Build comparison matrix
        W = np.zeros((n, n))
        for winner, loser, count in comparisons:
            i = system_to_idx[winner]
            j = system_to_idx[loser]
            W[i, j] += count
        
        # Maximum likelihood estimation
        def neg_log_likelihood(theta):
            exp_theta = np.exp(theta)
            likelihood = 0
            for i in range(n):
                for j in range(n):
                    if W[i, j] > 0:
                        likelihood += W[i, j] * (theta[i] - np.log(exp_theta[i] + exp_theta[j]))
            return -likelihood
        
        # Initialize and optimize
        initial_theta = np.zeros(n)
        result = minimize(neg_log_likelihood, initial_theta, method='BFGS')
        
        self.parameters = result.x - np.mean(result.x)  # Center parameters
        self.fitted = True
        
        logger.info(f"✅ Bradley-Terry model fitted for {n} systems")
    
    def get_elo_ratings(self, base_elo: float = 1500, scale: float = 400) -> Dict:
        """
        Convert Bradley-Terry parameters to Elo ratings.
        
        θ_i ≈ (Elo_i - base_elo) / scale
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        ratings = {}
        for i, system in enumerate(self.systems):
            ratings[system] = base_elo + self.parameters[i] * scale
        
        return ratings
    
    def predict_win_probability(self, system_a: str, system_b: str) -> float:
        """Predict probability that system_a beats system_b."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        idx_a = self.systems.index(system_a)
        idx_b = self.systems.index(system_b)
        
        theta_a = self.parameters[idx_a]
        theta_b = self.parameters[idx_b]
        
        return np.exp(theta_a) / (np.exp(theta_a) + np.exp(theta_b))
    
    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get ranking of systems by ability."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        rankings = [(self.systems[i], self.parameters[i]) for i in range(len(self.systems))]
        return sorted(rankings, key=lambda x: x[1], reverse=True)


class EloSystem:
    """
    Elo rating system for online preference updates.
    
    Used in Chatbot Arena-style evaluations.
    """
    
    def __init__(self, initial_elo: float = 1500, k_factor: float = 32):
        self.ratings = defaultdict(lambda: initial_elo)
        self.k_factor = k_factor
        self.history = []
    
    def update(self, winner: str, loser: str):
        """
        Update Elo ratings after a comparison.
        
        Args:
            winner: Winning system
            loser: Losing system
        """
        r_winner = self.ratings[winner]
        r_loser = self.ratings[loser]
        
        # Expected win probabilities
        expected_winner = 1 / (1 + 10 ** ((r_loser - r_winner) / 400))
        expected_loser = 1 / (1 + 10 ** ((r_winner - r_loser) / 400))
        
        # Update ratings
        self.ratings[winner] = r_winner + self.k_factor * (1 - expected_winner)
        self.ratings[loser] = r_loser + self.k_factor * (0 - expected_loser)
        
        self.history.append({
            "winner": winner,
            "loser": loser,
            "ratings": dict(self.ratings)
        })
    
    def get_ratings(self) -> Dict:
        """Get current ratings."""
        return dict(self.ratings)
    
    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get ranking by Elo rating."""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
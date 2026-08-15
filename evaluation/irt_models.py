"""
Item Response Theory Models for AI Evaluation

Includes:
- Rasch (1PL) Model
- 2PL Model
- 3PL Model
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import expit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaschModel:
    """
    Rasch (1PL) Model for binary responses.
    
    P(X_ij = 1 | θ_i, β_j) = exp(θ_i - β_j) / (1 + exp(θ_i - β_j))
    
    Where:
    - θ_i: Ability of system i
    - β_j: Difficulty of item j
    """
    
    def __init__(self, response_matrix):
        self.matrix = response_matrix
        self.abilities = None
        self.difficulties = None
        self.fitted = False
    
    def fit(self, max_iter: int = 100, tol: float = 1e-4):
        """Fit the Rasch model using Joint Maximum Likelihood."""
        logger.info("Fitting Rasch (1PL) Model...")
        
        n_systems = self.matrix.n_systems
        n_items = self.matrix.n_items
        
        # Initialize parameters
        abilities = np.zeros(n_systems)
        difficulties = np.zeros(n_items)
        
        for iteration in range(max_iter):
            # Update item difficulties
            for j in range(n_items):
                responses = self.matrix.matrix[:, j]
                valid = ~np.isnan(responses)
                if np.sum(valid) == 0:
                    continue
                
                # Newton-Raphson step for difficulty
                def item_likelihood(beta):
                    theta = abilities[valid]
                    p = expit(theta - beta)
                    return -np.sum(responses[valid] * np.log(p + 1e-10) + 
                                   (1 - responses[valid]) * np.log(1 - p + 1e-10))
                
                result = minimize(item_likelihood, difficulties[j], method='BFGS')
                difficulties[j] = result.x[0]
            
            # Update system abilities
            for i in range(n_systems):
                responses = self.matrix.matrix[i, :]
                valid = ~np.isnan(responses)
                if np.sum(valid) == 0:
                    continue
                
                def system_likelihood(theta):
                    beta = difficulties[valid]
                    p = expit(theta - beta)
                    return -np.sum(responses[valid] * np.log(p + 1e-10) + 
                                   (1 - responses[valid]) * np.log(1 - p + 1e-10))
                
                result = minimize(system_likelihood, abilities[i], method='BFGS')
                abilities[i] = result.x[0]
            
            # Check convergence
            if iteration > 0:
                if np.max(np.abs(abilities - prev_abilities)) < tol:
                    break
            
            prev_abilities = abilities.copy()
        
        self.abilities = abilities
        self.difficulties = difficulties
        self.fitted = True
        
        logger.info(f"✅ Rasch model fitted in {iteration+1} iterations")
    
    def get_item_fit(self) -> Dict:
        """Get item fit statistics (infit/outfit)."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        infit = []
        outfit = []
        
        for j in range(self.matrix.n_items):
            responses = self.matrix.matrix[:, j]
            valid = ~np.isnan(responses)
            
            if np.sum(valid) == 0:
                infit.append(np.nan)
                outfit.append(np.nan)
                continue
            
            theta = self.abilities[valid]
            beta = self.difficulties[j]
            p = expit(theta - beta)
            
            # Standardized residuals
            z = (responses[valid] - p) / np.sqrt(p * (1 - p) + 1e-10)
            
            infit.append(np.mean(z**2))
            outfit.append(np.mean(z**2 * (1 - p) * p))
        
        return {
            "items": self.matrix.items,
            "infit": infit,
            "outfit": outfit
        }
    
    def predict(self, ability: float, difficulty: float) -> float:
        """Predict probability of correct response."""
        return expit(ability - difficulty)
    
    def get_system_ability(self, system_idx: int) -> float:
        """Get ability estimate for a system."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        return self.abilities[system_idx]


class TwoPLModel:
    """
    2PL Model with discrimination parameter.
    
    P(X_ij = 1 | θ_i, α_j, β_j) = exp(α_j(θ_i - β_j)) / (1 + exp(α_j(θ_i - β_j)))
    
    Where:
    - θ_i: Ability of system i
    - β_j: Difficulty of item j
    - α_j: Discrimination of item j
    """
    
    def __init__(self, response_matrix):
        self.matrix = response_matrix
        self.abilities = None
        self.difficulties = None
        self.discriminations = None
        self.fitted = False
    
    def fit(self, max_iter: int = 100):
        """Fit the 2PL model."""
        logger.info("Fitting 2PL Model...")
        
        n_systems = self.matrix.n_systems
        n_items = self.matrix.n_items
        
        # Initialize parameters
        abilities = np.zeros(n_systems)
        difficulties = np.zeros(n_items)
        discriminations = np.ones(n_items)
        
        for iteration in range(max_iter):
            # Update item parameters
            for j in range(n_items):
                responses = self.matrix.matrix[:, j]
                valid = ~np.isnan(responses)
                if np.sum(valid) < 2:
                    continue
                
                theta = abilities[valid]
                
                def item_likelihood(params):
                    alpha, beta = params
                    p = expit(alpha * (theta - beta))
                    return -np.sum(responses[valid] * np.log(p + 1e-10) + 
                                   (1 - responses[valid]) * np.log(1 - p + 1e-10))
                
                result = minimize(item_likelihood, 
                                 [discriminations[j], difficulties[j]], 
                                 method='BFGS',
                                 bounds=[(0.01, 10), (-10, 10)])
                discriminations[j] = result.x[0]
                difficulties[j] = result.x[1]
            
            # Update system abilities
            for i in range(n_systems):
                responses = self.matrix.matrix[i, :]
                valid = ~np.isnan(responses)
                if np.sum(valid) == 0:
                    continue
                
                alpha = discriminations[valid]
                beta = difficulties[valid]
                
                def system_likelihood(theta):
                    p = expit(alpha * (theta - beta))
                    return -np.sum(responses[valid] * np.log(p + 1e-10) + 
                                   (1 - responses[valid]) * np.log(1 - p + 1e-10))
                
                result = minimize(system_likelihood, abilities[i], method='BFGS')
                abilities[i] = result.x[0]
            
            if iteration % 10 == 0:
                logger.debug(f"  Iteration {iteration} complete")
        
        self.abilities = abilities
        self.difficulties = difficulties
        self.discriminations = discriminations
        self.fitted = True
        
        logger.info("✅ 2PL model fitted")
    
    def get_item_info(self) -> Dict:
        """Get item parameters."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        return {
            "items": self.matrix.items,
            "difficulties": self.difficulties.tolist(),
            "discriminations": self.discriminations.tolist()
        }


class ThreePLModel:
    """
    3PL Model with guessing parameter.
    
    P(X_ij = 1 | θ_i, α_j, β_j, γ_j) = γ_j + (1 - γ_j) * exp(α_j(θ_i - β_j)) / (1 + exp(α_j(θ_i - β_j)))
    
    Where:
    - γ_j: Guessing parameter for item j
    """
    
    def __init__(self, response_matrix):
        self.matrix = response_matrix
        self.abilities = None
        self.difficulties = None
        self.discriminations = None
        self.guessing = None
        self.fitted = False
    
    def fit(self):
        """Fit the 3PL model (simplified)."""
        logger.info("Fitting 3PL Model (simplified)...")
        
        # First fit 2PL
        two_pl = TwoPLModel(self.matrix)
        two_pl.fit()
        
        # Then estimate guessing parameters
        n_items = self.matrix.n_items
        guessing = np.zeros(n_items)
        
        for j in range(n_items):
            scores = self.matrix.matrix[:, j]
            valid = ~np.isnan(scores)
            if np.sum(valid) == 0:
                continue
            
            # Guess = minimum score or 0.25 for 4-choice items
            min_score = np.min(scores[valid])
            guessing[j] = min(0.25, min_score)
        
        self.abilities = two_pl.abilities
        self.difficulties = two_pl.difficulties
        self.discriminations = two_pl.discriminations
        self.guessing = guessing
        self.fitted = True
        
        logger.info("✅ 3PL model fitted")


class IRTAnalyzer:
    """Analyze IRT model results."""
    
    def __init__(self, model):
        self.model = model
        self.matrix = model.matrix
    
    def item_characteristic_curves(self) -> Dict:
        """Generate item characteristic curves data."""
        if not self.model.fitted:
            raise ValueError("Model not fitted yet")
        
        theta_range = np.linspace(-3, 3, 100)
        curves = {}
        
        if isinstance(self.model, RaschModel):
            for j, item in enumerate(self.matrix.items):
                beta = self.model.difficulties[j]
                curves[item] = {
                    "theta": theta_range.tolist(),
                    "probability": expit(theta_range - beta).tolist(),
                    "difficulty": beta
                }
        
        return curves
    
    def test_information_function(self) -> Dict:
        """Compute test information function."""
        if not self.model.fitted:
            raise ValueError("Model not fitted yet")
        
        theta_range = np.linspace(-3, 3, 100)
        info = np.zeros_like(theta_range)
        
        if isinstance(self.model, RaschModel):
            for j in range(self.matrix.n_items):
                beta = self.model.difficulties[j]
                p = expit(theta_range - beta)
                info += p * (1 - p)
        
        return {
            "theta": theta_range.tolist(),
            "information": info.tolist()
        }
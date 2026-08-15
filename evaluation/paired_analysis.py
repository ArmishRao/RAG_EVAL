"""
Paired Response Matrix Analysis for Human-AI Collaboration

Measures:
- Human performance with vs without AI
- AI assistance effectiveness
- Learning outcomes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PairedResponseMatrix:
    """
    Paired Response Matrix for Human-AI collaboration studies.
    
    Contains:
    - Human-only responses
    - Human + AI responses
    - AI-only responses
    """
    items: List[str]
    human_only: np.ndarray
    human_ai: np.ndarray
    ai_only: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate dimensions."""
        n_items = len(self.items)
        if self.human_only.shape[0] != n_items:
            raise ValueError("Human-only shape mismatch")
        if self.human_ai.shape[0] != n_items:
            raise ValueError("Human-AI shape mismatch")
    
    def compute_ai_assistance_effect(self) -> Dict:
        """
        Compute the effect of AI assistance on human performance.
        
        Returns:
            - Improvement: Human+AI vs Human-only
            - Improvement percentage
            - Effect size
        """
        human_only_mean = np.nanmean(self.human_only)
        human_ai_mean = np.nanmean(self.human_ai)
        
        improvement = human_ai_mean - human_only_mean
        improvement_pct = (improvement / human_only_mean * 100) if human_only_mean > 0 else 0
        
        return {
            "human_only_mean": human_only_mean,
            "human_ai_mean": human_ai_mean,
            "improvement": improvement,
            "improvement_percentage": improvement_pct
        }
    
    def compute_ai_capability_gap(self) -> Dict:
        """
        Compute the gap between AI capability and human performance.
        
        Returns:
            - AI-human gap
            - Areas where AI outperforms humans
            - Areas where humans outperform AI
        """
        if self.ai_only is None:
            return {"error": "AI-only responses not provided"}
        
        gaps = self.ai_only - self.human_only
        mean_gap = np.nanmean(gaps)
        
        # Identify items where AI performs better
        ai_better = []
        human_better = []
        for i, item in enumerate(self.items):
            if not np.isnan(gaps[i]):
                if gaps[i] > 0:
                    ai_better.append({"item": item, "gap": gaps[i]})
                else:
                    human_better.append({"item": item, "gap": -gaps[i]})
        
        return {
            "mean_gap": mean_gap,
            "ai_better": len(ai_better),
            "human_better": len(human_better),
            "ai_better_items": ai_better[:5],  # Top 5
            "human_better_items": human_better[:5]  # Top 5
        }
    
    def compute_learning_effect(self, pre: np.ndarray, post: np.ndarray) -> Dict:
        """
        Compute learning effect from AI assistance.
        
        Args:
            pre: Pre-AI responses (human-only before)
            post: Post-AI responses (human-only after)
        """
        pre_mean = np.nanmean(pre)
        post_mean = np.nanmean(post)
        
        learning_gain = post_mean - pre_mean
        
        return {
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "learning_gain": learning_gain,
            "learning_percentage": (learning_gain / pre_mean * 100) if pre_mean > 0 else 0
        }
    
    def identify_collaboration_patterns(self) -> Dict:
        """
        Identify patterns in human-AI collaboration.
        
        Returns:
            - Complementary: AI helps where humans struggle
            - Redundant: AI and human similar
            - Misleading: AI hurts performance
        """
        patterns = {
            "complementary": [],
            "redundant": [],
            "misleading": []
        }
        
        for i, item in enumerate(self.items):
            human = self.human_only[i] if not np.isnan(self.human_only[i]) else 0
            ai = self.ai_only[i] if self.ai_only is not None and not np.isnan(self.ai_only[i]) else 0
            collab = self.human_ai[i] if not np.isnan(self.human_ai[i]) else 0
            
            if ai > human and collab > human:
                patterns["complementary"].append(item)
            elif abs(ai - human) < 0.1 and abs(collab - human) < 0.1:
                patterns["redundant"].append(item)
            elif ai < human and collab < human:
                patterns["misleading"].append(item)
        
        return patterns
    
    def summary(self) -> Dict:
        """Generate summary statistics."""
        summary = {
            "n_items": len(self.items),
            "human_only_mean": np.nanmean(self.human_only),
            "human_ai_mean": np.nanmean(self.human_ai),
        }
        
        if self.ai_only is not None:
            summary["ai_only_mean"] = np.nanmean(self.ai_only)
        
        # Assistance effect
        effect = self.compute_ai_assistance_effect()
        summary["assistance_effect"] = effect
        
        return summary
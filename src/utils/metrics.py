"""
Metrics calculation utilities for poker agent
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def compute_accuracy(predictions: List, targets: List) -> float:
    """
    Compute accuracy score
    
    Args:
        predictions: List of predicted values
        targets: List of target values
        
    Returns:
        Accuracy score (0-1)
    """
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def compute_top_k_accuracy(predictions: List[List], targets: List, k: int = 3) -> float:
    """
    Compute top-K accuracy
    
    Args:
        predictions: List of lists of top-K predictions
        targets: List of target values
        k: Number of top predictions to consider
        
    Returns:
        Top-K accuracy score
    """
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for preds, t in zip(predictions, targets) if t in preds[:k])
    return correct / len(predictions)


def compute_win_rate(wins: int, total: int) -> float:
    """
    Compute win rate
    
    Args:
        wins: Number of wins
        total: Total number of games/hands
        
    Returns:
        Win rate (0-1)
    """
    if total == 0:
        return 0.0
    return wins / total


def compute_confusion_matrix(
    predictions: List,
    targets: List,
    labels: Optional[List] = None
) -> Dict[str, Any]:
    """
    Compute confusion matrix and classification metrics
    
    Args:
        predictions: List of predicted values
        targets: List of target values
        labels: List of label names
        
    Returns:
        Dictionary with confusion matrix and metrics
    """
    if len(predictions) == 0:
        return {'error': 'No predictions'}
    
    # Get unique labels
    if labels is None:
        labels = sorted(set(targets + predictions))
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(len(labels)))
    
    # Compute per-class metrics
    class_metrics = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[str(label)] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': tp + fn
        }
    
    # Overall metrics
    accuracy = accuracy_score(targets, predictions)
    
    return {
        'confusion_matrix': cm.tolist(),
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'labels': labels
    }


class MetricsTracker:
    """
    Track and compute various metrics during training/evaluation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics"""
        self.actions = []
        self.rewards = []
        self.stacks = []
        self.pots = []
        self.hand_results = []
    
    def log_action(self, action: str, reward: float, stack: float, pot: float):
        """Log an action"""
        self.actions.append(action)
        self.rewards.append(reward)
        self.stacks.append(stack)
        self.pots.append(pot)
    
    def log_hand_result(self, result: dict):
        """Log hand result"""
        self.hand_results.append(result)
    
    def get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions"""
        distribution = defaultdict(int)
        for action in self.actions:
            distribution[action] += 1
        return dict(distribution)
    
    def get_action_proportions(self) -> Dict[str, float]:
        """Get proportions of actions"""
        total = len(self.actions)
        if total == 0:
            return {}
        
        distribution = self.get_action_distribution()
        return {k: v / total for k, v in distribution.items()}
    
    def get_average_reward(self) -> float:
        """Get average reward per action"""
        if len(self.rewards) == 0:
            return 0.0
        return np.mean(self.rewards)
    
    def get_total_reward(self) -> float:
        """Get total reward"""
        return sum(self.rewards)
    
    def get_win_rate(self) -> float:
        """Get win rate from hand results"""
        if len(self.hand_results) == 0:
            return 0.0
        
        wins = sum(1 for r in self.hand_results if r.get('is_win', False))
        return wins / len(self.hand_results)
    
    def get_average_profit(self) -> float:
        """Get average profit per hand"""
        if len(self.hand_results) == 0:
            return 0.0
        
        profits = [r.get('profit', 0) for r in self.hand_results]
        return np.mean(profits)
    
    def get_volatility(self) -> float:
        """Get reward volatility (standard deviation)"""
        if len(self.rewards) == 0:
            return 0.0
        return np.std(self.rewards)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            'num_actions': len(self.actions),
            'num_hands': len(self.hand_results),
            'action_distribution': self.get_action_distribution(),
            'action_proportions': self.get_action_proportions(),
            'average_reward': self.get_average_reward(),
            'total_reward': self.get_total_reward(),
            'win_rate': self.get_win_rate(),
            'average_profit': self.get_average_profit(),
            'volatility': self.get_volatility()
        }


def compute_expected_value(results: List[float]) -> float:
    """
    Compute expected value from results
    
    Args:
        results: List of outcomes (positive for wins, negative for losses)
        
    Returns:
        Expected value
    """
    if len(results) == 0:
        return 0.0
    return np.mean(results)


def compute_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio for returns
    
    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (default 0)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def compute_max_drawdown(profits: List[float]) -> float:
    """
    Compute maximum drawdown from profit sequence
    
    Args:
        profits: List of cumulative or per-hand profits
        
    Returns:
        Maximum drawdown (positive value)
    """
    if len(profits) == 0:
        return 0.0
    
    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)
    
    return abs(max_drawdown)
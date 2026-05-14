"""
Training Module - Supervised, RL, Self-Play, and Evaluation
"""

# Import from supervised
from .supervised import SupervisedTrainer, SupervisedConfig, train_supervised

# These will be added later when files are created
# from .rl_trainer import PPOConfig, PPOTrainer
# from .self_play import SelfPlayTrainer, SelfPlayConfig
# from .evaluation import Evaluator, EvaluationConfig

__all__ = [
    'SupervisedTrainer',
    'SupervisedConfig',
    'train_supervised',
    # 'PPOConfig',
    # 'PPOTrainer',
    # 'SelfPlayTrainer',
    # 'SelfPlayConfig',
    # 'Evaluator',
    # 'EvaluationConfig'
]
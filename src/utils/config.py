"""
Configuration management for poker agent
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Config:
    """Main configuration class for poker agent"""
    
    # Environment settings
    env_name: str = "PokerEnv"
    num_players: int = 6
    starting_stack: float = 1000.0
    small_blind: float = 5.0
    big_blind: float = 10.0
    
    # Agent settings
    agent_type: str = "policy"
    agent_name: str = "PokerAgent"
    
    # Model settings
    model_input_dim: int = 115
    model_hidden_dims: list = field(default_factory=lambda: [256, 128])
    model_output_dim: int = 6
    model_dropout: float = 0.3
    
    # Training settings
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 50
    device: str = "cpu"
    
    # PPO settings
    ppo_clip_epsilon: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    
    # Self-play settings
    self_play_iterations: int = 100
    self_play_hands_per_iter: int = 500
    self_play_update_freq: int = 5
    
    # Evaluation settings
    eval_num_hands: int = 1000
    eval_num_opponents: int = 5
    
    # Data settings
    data_dir: str = "data"
    processed_dir: str = "data/processed"
    raw_dir: str = "data/raw"
    
    # Experiment settings
    experiment_name: str = ""
    checkpoint_dir: str = "experiments/checkpoints"
    log_dir: str = "experiments/logs"
    results_dir: str = "experiments/results"
    
    # Logging settings
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "poker-agent"
    
    # API settings
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-3.5-turbo"
    llm_use_local: bool = False
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'env_name': self.env_name,
            'num_players': self.num_players,
            'starting_stack': self.starting_stack,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'agent_type': self.agent_type,
            'agent_name': self.agent_name,
            'model_input_dim': self.model_input_dim,
            'model_hidden_dims': self.model_hidden_dims,
            'model_output_dim': self.model_output_dim,
            'model_dropout': self.model_dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'device': self.device,
            'ppo_clip_epsilon': self.ppo_clip_epsilon,
            'ppo_value_coef': self.ppo_value_coef,
            'ppo_entropy_coef': self.ppo_entropy_coef,
            'ppo_gamma': self.ppa_gamma,
            'ppo_gae_lambda': self.ppo_gae_lambda,
            'self_play_iterations': self.self_play_iterations,
            'self_play_hands_per_iter': self.self_play_hands_per_iter,
            'self_play_update_freq': self.self_play_update_freq,
            'eval_num_hands': self.eval_num_hands,
            'eval_num_opponents': self.eval_num_opponents,
            'data_dir': self.data_dir,
            'processed_dir': self.processed_dir,
            'raw_dir': self.raw_dir,
            'experiment_name': self.experiment_name,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'results_dir': self.results_dir,
            'log_level': self.log_level,
            'use_wandb': self.use_wandb,
            'wandb_project': self.wandb_project,
            'llm_api_key': self.llm_api_key,
            'llm_model': self.llm_model,
            'llm_use_local': self.llm_use_local,
            'seed': self.seed
        }
    
    def to_yaml(self, filepath: str):
        """Save config to YAML file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, filepath: str):
        """Save config to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """Load config from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}")
        print("="*50 + "\n")


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default
    
    Args:
        config_path: Path to config file (YAML or JSON)
        
    Returns:
        Config object
    """
    if config_path is None:
        return Config()
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"⚠️ Config file not found: {config_path}, using defaults")
        return Config()
    
    if config_path.suffix in ['.yaml', '.yml']:
        return Config.from_yaml(config_path)
    elif config_path.suffix == '.json':
        return Config.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Config, config_path: str):
    """
    Save configuration to file
    
    Args:
        config: Config object
        config_path: Path to save config
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix in ['.yaml', '.yml']:
        config.to_yaml(config_path)
    elif config_path.suffix == '.json':
        config.to_json(config_path)
    else:
        config.to_yaml(config_path.with_suffix('.yaml'))


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """
    Update configuration with dictionary
    
    Args:
        config: Original config
        updates: Dictionary of updates
        
    Returns:
        Updated config
    """
    config.update(updates)
    return config
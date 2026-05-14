#!/usr/bin/env python3
"""
Script 4: Train RL agent using PPO
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.environment.poker_env import PokerEnv, PokerEnvConfig
from src.models.policy_net import PolicyNetwork
from src.models.value_net import ValueNetwork
from src.training.rl_trainer import PPOTrainer, PPOConfig
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train RL poker agent')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes')
    parser.add_argument('--num-opponents', type=int, default=5,
                        help='Number of opponents')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained supervised model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--algorithm', type=str, default='ppo',
                        help='RL algorithm (ppo, a2c)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"RL Training - {args.algorithm.upper()}")
    print("=" * 60)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create environment
    env_config = PokerEnvConfig(
        num_players=args.num_opponents + 1,
        starting_stack=1000.0,
        small_blind=5.0,
        big_blind=10.0
    )
    env = PokerEnv(env_config)
    print(f"Environment: {args.num_opponents} opponents, {env_config.starting_stack} chips")
    
    # Create networks
    print("\n🏗️  Creating networks...")
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    
    # Load pretrained model if provided
    if args.pretrained and Path(args.pretrained).exists():
        print(f"Loading pretrained model from {args.pretrained}")
        policy_net.load_state_dict(torch.load(args.pretrained, map_location=device))
    
    print(f"   Policy network params: {sum(p.numel() for p in policy_net.parameters()):,}")
    print(f"   Value network params: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # Training config
    train_config = PPOConfig(
        num_episodes=args.episodes,
        num_opponents=args.num_opponents,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
        log_interval=100,
        save_interval=500,
        use_wandb=False
    )
    
    # Create trainer
    print(f"\n🚀 Starting {args.algorithm.upper()} training for {args.episodes} episodes...")
    print("-" * 40)
    
    trainer = PPOTrainer(env, policy_net, value_net, train_config, device)
    results = trainer.train()
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Average reward (last 100 episodes): {results['avg_reward_last_100']:.2f}")
    print(f"Best episode reward: {max(results['episode_rewards']):.2f}")
    
    print(f"\n✅ Model saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
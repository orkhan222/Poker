#!/usr/bin/env python3
"""
Script 6: Run self-play training
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json

from src.training.self_play import SelfPlayTrainer, SelfPlayConfig
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Run self-play training')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of self-play iterations')
    parser.add_argument('--hands-per-iter', type=int, default=500,
                        help='Hands per iteration')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model')
    parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Self-Play Training")
    print("=" * 60)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")
    print(f"Hands per iteration: {args.hands_per_iter}")
    
    # Load pretrained model if provided
    if args.pretrained and Path(args.pretrained).exists():
        print(f"Loading pretrained model from: {args.pretrained}")
    
    # Self-play config
    config = SelfPlayConfig(
        num_iterations=args.iterations,
        hands_per_iteration=args.hands_per_iter,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
        save_interval=5,
        eval_interval=5,
        use_wandb=False
    )
    
    # Create trainer
    print("\n🚀 Starting self-play training...")
    print("-" * 40)
    
    trainer = SelfPlayTrainer(config, device)
    results = trainer.train(load_checkpoint=args.resume)
    
    # Print results
    print("\n" + "=" * 60)
    print("Self-Play Complete!")
    print("=" * 60)
    
    # Print win rate progression
    win_rates = results['win_rates']
    if win_rates:
        print(f"Initial win rate: {win_rates[0]:.3f}")
        print(f"Final win rate: {win_rates[-1]:.3f}")
        print(f"Improvement: {(win_rates[-1] - win_rates[0])*100:.1f}%")
    
    # Print ELO progression
    elos = results['elos']
    if elos:
        print(f"Initial ELO: {elos[0]:.0f}")
        print(f"Final ELO: {elos[-1]:.0f}")
        print(f"ELO gain: {elos[-1] - elos[0]:.0f}")
    
    # Final evaluation
    if results.get('final_eval'):
        print(f"\nFinal average win rate: {results['final_eval'].get('avg_win_rate', 0):.3f}")
    
    print(f"\n✅ Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script 5: Evaluate trained model against baselines
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json

from src.agents.policy_agent import PolicyAgent
from src.agents.rule_agent import RuleBasedAgent, RandomAgent
from src.training.evaluation import Evaluator, EvaluationConfig
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Evaluate poker agent')
    parser.add_argument('--model', type=str, default='experiments/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num-hands', type=int, default=1000,
                        help='Number of hands per evaluation')
    parser.add_argument('--output', type=str, default='experiments/results/evaluation.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load agent
    print(f"\n📂 Loading model from: {args.model}")
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    agent = PolicyAgent(model_path=args.model, device=device, name="TrainedAgent")
    print(f"✅ Agent loaded: {agent.name}")
    
    # Evaluation config
    eval_config = EvaluationConfig(
        num_hands=args.num_hands,
        verbose=args.verbose
    )
    evaluator = Evaluator(eval_config)
    
    # Run evaluation
    print(f"\n🚀 Evaluating against baselines ({args.num_hands} hands each)...")
    print("-" * 40)
    
    results = evaluator.evaluate_against_baselines(agent, num_hands=args.num_hands)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    for opponent, metrics in results.items():
        print(f"\nvs {opponent}:")
        print(f"  Win Rate: {metrics['win_rate']:.3f} ({metrics['win_rate']*100:.1f}%)")
        print(f"  Avg Profit: ${metrics['avg_profit']:.2f}")
        print(f"  Total Profit: ${metrics['total_profit']:.2f}")
    
    # Calculate average
    win_rates = [m['win_rate'] for m in results.values()]
    avg_win_rate = sum(win_rates) / len(win_rates)
    print(f"\n{'='*40}")
    print(f"Average Win Rate: {avg_win_rate:.3f} ({avg_win_rate*100:.1f}%)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'model': args.model,
            'num_hands': args.num_hands,
            'results': results,
            'avg_win_rate': avg_win_rate
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
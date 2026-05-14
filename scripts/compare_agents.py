#!/usr/bin/env python3
"""
Script 7: Compare multiple agents against each other
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd

from src.agents.policy_agent import PolicyAgent
from src.agents.rule_agent import RuleBasedAgent, RandomAgent
from src.agents.llm_agent import LLMAgent, LLMAgentConfig
from src.training.evaluation import Evaluator, EvaluationConfig
from src.utils.logger import setup_logger


def create_agent(agent_type, name=None, model_path=None, device='cpu'):
    """Create agent by type"""
    if agent_type == 'policy':
        return PolicyAgent(name=name or 'Policy', model_path=model_path, device=device)
    elif agent_type == 'conservative':
        return RuleBasedAgent(name=name or 'Conservative', strategy='conservative')
    elif agent_type == 'aggressive':
        return RuleBasedAgent(name=name or 'Aggressive', strategy='aggressive')
    elif agent_type == 'tight':
        return RuleBasedAgent(name=name or 'Tight', strategy='tight')
    elif agent_type == 'loose':
        return RuleBasedAgent(name=name or 'Loose', strategy='loose')
    elif agent_type == 'random':
        return RandomAgent(name=name or 'Random')
    elif agent_type == 'llm':
        config = LLMAgentConfig(use_local=True)
        return LLMAgent(name=name or 'LLM', config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser(description='Compare agents')
    parser.add_argument('--agents', type=str, nargs='+',
                        default=['conservative', 'aggressive', 'tight', 'loose', 'random'],
                        help='Agent types to compare')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained policy model')
    parser.add_argument('--num-hands', type=int, default=1000,
                        help='Number of hands per pair')
    parser.add_argument('--output', type=str, default='experiments/results/comparison.csv',
                        help='Output CSV file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Agent Comparison Tournament")
    print("=" * 60)
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Hands per pair: {args.num_hands}")
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Create agents
    print("\n📂 Creating agents...")
    agents = []
    names = []
    
    for agent_type in args.agents:
        if agent_type == 'policy' and args.model:
            name = f"Policy (trained)"
        else:
            name = agent_type.capitalize()
        
        agent = create_agent(agent_type, name, args.model, device)
        agents.append(agent)
        names.append(name)
        print(f"   {name} ({agent_type})")
    
    # Create evaluator
    eval_config = EvaluationConfig(num_hands=args.num_hands, verbose=False)
    evaluator = Evaluator(eval_config)
    
    # Run round-robin tournament
    print(f"\n🚀 Running round-robin tournament ({len(agents)} agents)...")
    print("-" * 40)
    
    results_df = evaluator.compare_agents(agents, names, num_hands=args.num_hands)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path)
    
    print(f"\n✅ Results saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
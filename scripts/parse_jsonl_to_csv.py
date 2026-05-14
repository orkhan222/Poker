#!/usr/bin/env python3
"""
Script 2: Preprocess and clean data
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Preprocess poker data')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='data/datasets',
                        help='Directory to save processed datasets')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Check if CSV files exist
    data_dir = Path(args.data_dir)
    required_files = ['hands.csv', 'players.csv', 'actions.csv', 'stack_events.csv']
    
    missing_files = []
    for f in required_files:
        if not (data_dir / f).exists():
            missing_files.append(f)
    
    if missing_files:
        print(f"\n❌ Missing CSV files in {args.data_dir}: {missing_files}")
        print("Please run the data generation script first:")
        print("   python create_and_parse.py")
        sys.exit(1)
    
    # Load data
    print("\n📂 Loading CSV files...")
    hands = pd.read_csv(data_dir / 'hands.csv')
    players = pd.read_csv(data_dir / 'players.csv')
    actions = pd.read_csv(data_dir / 'actions.csv')
    stack_events = pd.read_csv(data_dir / 'stack_events.csv')
    
    print(f"   Hands: {len(hands)}")
    print(f"   Players: {len(players)}")
    print(f"   Actions: {len(actions)}")
    print(f"   Stack Events: {len(stack_events)}")
    
    # Basic preprocessing
    print("\n🧹 Basic preprocessing...")
    
    # Fill missing values
    if 'cards' in players.columns:
        players['cards'] = players['cards'].fillna('')
    if 'starting_stack' in players.columns:
        players['starting_stack'] = players['starting_stack'].fillna(1000)
    if 'ending_stack' in players.columns:
        players['ending_stack'] = players['ending_stack'].fillna(1000)
    
    # Create action encoding
    if 'action' in actions.columns:
        action_map = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}
        actions['action_encoded'] = actions['action'].map(action_map).fillna(0).astype(int)
    
    # Create street encoding
    if 'street' in actions.columns:
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        actions['street_encoded'] = actions['street'].map(street_map).fillna(0).astype(int)
    
    # Simple hand strength feature
    if 'cards' in players.columns:
        def simple_hand_strength(cards):
            if not cards or pd.isna(cards):
                return 0
            high_cards = {'A': 14, 'K': 13, 'Q': 12, 'J': 11}
            ranks = [c[0] for c in str(cards).split() if c]
            strength = 0
            for r in ranks:
                if r in high_cards:
                    strength += high_cards[r] / 14
                elif r.isdigit():
                    strength += int(r) / 14
            return min(strength / len(ranks) if ranks else 0, 1.0)
        
        players['hand_strength'] = players['cards'].apply(simple_hand_strength)
    
    # Normalize stacks
    if 'starting_stack' in players.columns:
        players['starting_stack_norm'] = players['starting_stack'] / 10000
        players['starting_stack_norm'] = players['starting_stack_norm'].clip(0, 1)
    
    # Save processed data
    print("\n💾 Saving processed data...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hands.to_csv(output_dir / 'hands_clean.csv', index=False)
    players.to_csv(output_dir / 'players_clean.csv', index=False)
    actions.to_csv(output_dir / 'actions_clean.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"   Cleaned data saved to: {args.output_dir}")
    
    # Action distribution
    if 'action' in actions.columns:
        print("\n📊 Action Distribution:")
        dist = actions['action'].value_counts()
        for action, count in dist.items():
            print(f"   {action}: {count} ({count/len(actions)*100:.1f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
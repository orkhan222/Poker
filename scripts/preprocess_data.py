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

from src.data.loader import load_all_data
from src.data.preprocess import DataPreprocessor


def main():
    parser = argparse.ArgumentParser(description='Preprocess poker data')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='data/datasets',
                        help='Directory to save processed datasets')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for testing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading CSV files...")
    try:
        hands, players, actions, stack_events = load_all_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run parse_jsonl_to_csv.py first")
        sys.exit(1)
    
    # Sample if needed
    if args.sample_size:
        print(f"⚠️ Using sample size: {args.sample_size}")
        hands = hands.head(args.sample_size)
        players = players.head(args.sample_size * 6)
        actions = actions.head(args.sample_size * 20)
    
    # Initial stats
    print("\n📊 Initial Data Statistics:")
    print(f"   Hands: {len(hands)}")
    print(f"   Players: {len(players)}")
    print(f"   Actions: {len(actions)}")
    
    # Preprocess
    print("\n🧹 Cleaning data...")
    preprocessor = DataPreprocessor()
    
    # Fit on data
    preprocessor.fit(actions, players)
    
    # Transform
    hands_clean, players_clean, actions_clean, stack_clean = preprocessor.preprocess(
        hands, players, actions, stack_events
    )
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Cleaned actions: {len(actions_clean)}")
    
    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hands_clean.to_csv(output_dir / 'hands_clean.csv', index=False)
    players_clean.to_csv(output_dir / 'players_clean.csv', index=False)
    actions_clean.to_csv(output_dir / 'actions_clean.csv', index=False)
    
    print(f"\n✅ Cleaned data saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
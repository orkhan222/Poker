#!/usr/bin/env python3
"""
Script 9: Exploratory Data Analysis (EDA) for poker dataset
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.loader import load_all_data
from src.utils.logger import setup_logger


def setup_plot_style():
    """Setup matplotlib style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_action_distribution(actions_df, output_dir):
    """Plot action distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Overall action distribution
    if 'action' in actions_df.columns:
        action_counts = actions_df['action'].value_counts()
        axes[0].bar(action_counts.index, action_counts.values)
        axes[0].set_title('Action Distribution (Overall)')
        axes[0].set_xlabel('Action')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (action, count) in enumerate(action_counts.items()):
            axes[0].text(i, count + max(action_counts.values)*0.01, 
                        f'{count:,}', ha='center', fontsize=10)
    
    # Action distribution by street
    if 'action' in actions_df.columns and 'street' in actions_df.columns:
        cross_tab = pd.crosstab(actions_df['street'], actions_df['action'])
        cross_tab.plot(kind='bar', ax=axes[1], stacked=True)
        axes[1].set_title('Action Distribution by Street')
        axes[1].set_xlabel('Street')
        axes[1].set_ylabel('Count')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: action_distribution.png")


def plot_stack_distribution(players_df, output_dir):
    """Plot stack distribution"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Starting stack distribution
    if 'starting_stack' in players_df.columns:
        axes[0].hist(players_df['starting_stack'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title('Starting Stack Distribution')
        axes[0].set_xlabel('Stack Size')
        axes[0].set_ylabel('Frequency')
    
    # Stack delta (profit/loss)
    if 'stack_delta' in players_df.columns:
        axes[1].hist(players_df['stack_delta'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_title('Profit/Loss Distribution')
        axes[1].set_xlabel('Stack Delta')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Hand strength vs stack delta
    if 'hand_strength' in players_df.columns and 'stack_delta' in players_df.columns:
        # Bin hand strength
        players_df['strength_bin'] = pd.cut(players_df['hand_strength'], bins=10)
        avg_profit_by_strength = players_df.groupby('strength_bin')['stack_delta'].mean()
        
        axes[2].bar(range(len(avg_profit_by_strength)), avg_profit_by_strength.values)
        axes[2].set_title('Average Profit by Hand Strength')
        axes[2].set_xlabel('Hand Strength (deciles)')
        axes[2].set_ylabel('Average Profit')
        axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stack_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: stack_distribution.png")


def plot_position_analysis(players_df, actions_df, output_dir):
    """Plot position-based analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Position win rate
    if 'position' in players_df.columns and 'stack_delta' in players_df.columns:
        pos_profit = players_df.groupby('position')['stack_delta'].mean().sort_values(ascending=False)
        axes[0].bar(pos_profit.index, pos_profit.values)
        axes[0].set_title('Average Profit by Position')
        axes[0].set_xlabel('Position')
        axes[0].set_ylabel('Average Profit')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        # Add value labels
        for i, (pos, profit) in enumerate(pos_profit.items()):
            axes[0].text(i, profit + (max(pos_profit.values)*0.02 if profit > 0 else min(pos_profit.values)*0.02),
                        f'${profit:.0f}', ha='center', fontsize=9)
    
    # Position action distribution
    if 'position' in actions_df.columns and 'action' in actions_df.columns:
        # Get top 6 positions
        top_positions = actions_df['position'].value_counts().head(6).index
        pos_actions = actions_df[actions_df['position'].isin(top_positions)]
        cross_tab = pd.crosstab(pos_actions['position'], pos_actions['action'], normalize='index')
        cross_tab.plot(kind='bar', ax=axes[1], stacked=True)
        axes[1].set_title('Action Distribution by Position')
        axes[1].set_xlabel('Position')
        axes[1].set_ylabel('Proportion')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: position_analysis.png")


def plot_correlation_heatmap(players_df, output_dir):
    """Plot correlation heatmap"""
    numeric_cols = players_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = players_df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: correlation_heatmap.png")


def generate_report(hands_df, players_df, actions_df, output_dir):
    """Generate summary report"""
    report = []
    report.append("=" * 60)
    report.append("POKER DATASET ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic statistics
    report.append("📊 BASIC STATISTICS")
    report.append("-" * 40)
    report.append(f"Total hands: {len(hands_df):,}")
    report.append(f"Total player records: {len(players_df):,}")
    report.append(f"Total actions: {len(actions_df):,}")
    report.append("")
    
    # Hand statistics
    if 'total_actions' in hands_df.columns:
        report.append("📈 HAND STATISTICS")
        report.append("-" * 40)
        report.append(f"Average actions per hand: {hands_df['total_actions'].mean():.1f}")
        report.append(f"Max actions per hand: {hands_df['total_actions'].max()}")
        report.append(f"Min actions per hand: {hands_df['total_actions'].min()}")
        report.append("")
    
    # Action statistics
    if 'action' in actions_df.columns:
        report.append("🎯 ACTION STATISTICS")
        report.append("-" * 40)
        action_counts = actions_df['action'].value_counts()
        for action, count in action_counts.items():
            report.append(f"{action}: {count:,} ({count/len(actions_df)*100:.1f}%)")
        report.append("")
    
    # Street statistics
    if 'street' in actions_df.columns:
        report.append("🃏 STREET STATISTICS")
        report.append("-" * 40)
        street_counts = actions_df['street'].value_counts()
        for street, count in street_counts.items():
            report.append(f"{street}: {count:,} ({count/len(actions_df)*100:.1f}%)")
        report.append("")
    
    # Position statistics
    if 'position' in players_df.columns:
        report.append("💺 POSITION STATISTICS")
        report.append("-" * 40)
        pos_counts = players_df['position'].value_counts()
        for pos, count in pos_counts.head(10).items():
            report.append(f"{pos}: {count:,}")
        report.append("")
    
    # Profit statistics
    if 'stack_delta' in players_df.columns:
        report.append("💰 PROFIT STATISTICS")
        report.append("-" * 40)
        report.append(f"Average profit per hand: ${players_df['stack_delta'].mean():.2f}")
        report.append(f"Median profit per hand: ${players_df['stack_delta'].median():.2f}")
        report.append(f"Max profit: ${players_df['stack_delta'].max():.2f}")
        report.append(f"Min profit: ${players_df['stack_delta'].min():.2f}")
        report.append(f"Standard deviation: ${players_df['stack_delta'].std():.2f}")
        report.append("")
    
    report.append("=" * 60)
    
    # Save report
    report_path = output_dir / 'eda_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"   Saved: eda_report.txt")
    return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='experiments/results/eda',
                        help='Directory to save plots and reports')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Exploratory Data Analysis (EDA)")
    print("=" * 60)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    
    # Load data
    print("\n📂 Loading data...")
    hands, players, actions, stack_events = load_all_data(args.data_dir)
    
    print(f"\n✅ Data loaded:")
    print(f"   Hands: {len(hands):,}")
    print(f"   Players: {len(players):,}")
    print(f"   Actions: {len(actions):,}")
    print(f"   Stack Events: {len(stack_events):,}")
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    
    plot_action_distribution(actions, output_dir)
    plot_stack_distribution(players, output_dir)
    plot_position_analysis(players, actions, output_dir)
    
    if len(players.select_dtypes(include=[np.number]).columns) > 1:
        plot_correlation_heatmap(players, output_dir)
    
    # Generate report
    print("\n📝 Generating report...")
    report = generate_report(hands, players, actions, output_dir)
    
    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("\n" + report)
    print("=" * 60)


if __name__ == "__main__":
    main()
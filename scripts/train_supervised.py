#!/usr/bin/env python3
"""
Script 3: Train supervised model on human poker data
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from src.data.loader import load_all_data
from src.data.preprocess import DataPreprocessor
from src.data.features import PokerFeatureExtractor
from src.data.dataset import PokerDataset
from src.models.policy_net import PolicyNetwork, PolicyNetworkConfig
from src.training.supervised import SupervisedTrainer, SupervisedConfig
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Train supervised poker model')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='experiments/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Sample size for testing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Supervised Training - Policy Network")
    print("=" * 60)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create sample data if real data doesn't exist
    from src.data.dataset import create_sample_dataset
    players_df, actions_df = create_sample_dataset(args.sample_size)
    
    # Create dataset
    print("\n📊 Creating dataset...")
    feature_extractor = PokerFeatureExtractor()
    dataset = PokerDataset(players_df, actions_df, feature_extractor)
    print(f"   Total samples: {len(dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Batch size: {batch_size}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model_config = PolicyNetworkConfig(
        input_dim=115,
        hidden_dims=[256, 128],
        output_dim=6,
        dropout_rate=0.3
    )
    model = PolicyNetwork(model_config)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    train_config = SupervisedConfig(
        batch_size=batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.output_dir,
        device=str(device),
        early_stopping_patience=5
    )
    
    # Train
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("-" * 40)
    
    trainer = SupervisedTrainer(model, train_config, device)
    results = trainer.train(train_loader, val_loader, test_loader)
    
    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    
    if results['test_metrics']:
        print(f"Test accuracy: {results['test_metrics']['accuracy']:.2f}%")
        print(f"Test loss: {results['test_metrics']['loss']:.4f}")
    
    print(f"\n✅ Model saved to: {args.output_dir}/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
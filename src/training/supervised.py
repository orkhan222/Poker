"""
Supervised learning pretraining on historical poker data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path


@dataclass
class SupervisedConfig:
    """Configuration for supervised training"""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    scheduler: str = 'step'
    weight_decay: float = 0.0
    dropout_rate: float = 0.3
    label_smoothing: float = 0.0
    checkpoint_dir: str = 'experiments/checkpoints'
    log_interval: int = 100
    eval_interval: int = 1
    early_stopping_patience: int = 10
    device: str = 'cpu'


class SupervisedTrainer:
    """
    Trainer for supervised learning on human poker data.
    Trains policy network to mimic human actions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SupervisedConfig = None,
        device: str = 'cpu'
    ):
        self.model = model
        self.config = config or SupervisedConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        
        # Optimizer
        if self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        
        # Scheduler
        if self.config.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=20, 
                gamma=0.5
            )
        elif self.config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        else:
            self.scheduler = None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, targets in tqdm(val_loader, desc='Validating', leave=False):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(features)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"SUPERVISED TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        if test_loader:
            print(f"Test samples: {len(test_loader.dataset)}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])
                
                # Print metrics
                print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
                print(f"Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val Acc: {val_metrics['accuracy']:.2f}%")
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.patience_counter = 0
                    
                    # Save checkpoint
                    checkpoint_path = Path(self.config.checkpoint_dir) / 'best_model.pt'
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"  ✅ Best model saved to {checkpoint_path}")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement for {self.patience_counter} epochs")
                    
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                    break
            else:
                print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
                print(f"Train Acc: {train_metrics['accuracy']:.2f}%")
        
        # Final evaluation on test set
        test_metrics = None
        if test_loader:
            test_metrics = self.validate(test_loader)
            print(f"\n{'='*40}")
            print(f"TEST SET RESULTS")
            print(f"{'='*40}")
            print(f"Test Loss: {test_metrics['loss']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        
        return {
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'test_metrics': test_metrics
        }


def train_supervised(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    model: Optional[nn.Module] = None,
    config: Optional[SupervisedConfig] = None,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to train supervised model
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        model: Model to train (creates new if None)
        config: Training configuration
        device: Device to use
        
    Returns:
        (trained_model, results) tuple
    """
    from src.models.policy_net import PolicyNetwork
    
    if model is None:
        model = PolicyNetwork()
    
    if config is None:
        config = SupervisedConfig()
    
    trainer = SupervisedTrainer(model, config, device)
    results = trainer.train(train_loader, val_loader, test_loader)
    
    return model, results


# For direct testing
if __name__ == "__main__":
    print("Supervised training module loaded successfully!")
    print(f"SupervisedTrainer: {SupervisedTrainer}")
    print(f"SupervisedConfig: {SupervisedConfig}")
    print(f"train_supervised: {train_supervised}")
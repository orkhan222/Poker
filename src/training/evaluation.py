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
import wandb
from pathlib import Path

from src.models.policy_net import PolicyNetwork
from src.models.utils import create_optimizer, create_scheduler, ModelCheckpoint, EarlyStopping


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
    class_weights: Optional[List[float]] = None
    use_wandb: bool = False
    wandb_project: str = 'poker-agent'
    checkpoint_dir: str = 'experiments/checkpoints'
    log_interval: int = 100
    eval_interval: int = 1
    early_stopping_patience: int = 10
    device: str = 'cuda'


class SupervisedTrainer:
    """
    Trainer for supervised learning on human poker data.
    Trains policy network to mimic human actions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SupervisedConfig = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config or SupervisedConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(self.device)
        
        # Loss function
        if self.config.class_weights:
            class_weights = torch.tensor(self.config.class_weights).to(self.device)
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=self.config.label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        
        # Optimizer
        self.optimizer = create_optimizer(
            self.model,
            self.config.optimizer,
            self.config.learning_rate,
            self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config.scheduler,
            step_size=20,
            gamma=0.5
        )
        
        # Checkpoint manager
        self.checkpointer = ModelCheckpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            mode='max'
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        
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
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    'train_batch_loss': loss.item(),
                    'train_batch_accuracy': 100. * correct / total,
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
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
        
        # Initialize wandb
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                    'optimizer': self.config.optimizer,
                    'model': self.model.__class__.__name__
                }
            )
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            
            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader, epoch)
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
                
                # Save checkpoint
                if val_metrics['accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.checkpointer.save(
                        self.model,
                        self.optimizer,
                        epoch,
                        {'val_accuracy': val_metrics['accuracy'], 'val_loss': val_metrics['loss']},
                        'supervised'
                    )
                
                # Early stopping
                if self.early_stopping.update(val_metrics['accuracy']):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
            else:
                print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
                print(f"Train Acc: {train_metrics['accuracy']:.2f}%")
        
        # Final evaluation on test set
        test_metrics = None
        if test_loader:
            test_metrics = self.validate(test_loader, epoch=0)
            print(f"\n{'='*40}")
            print(f"TEST SET RESULTS")
            print(f"{'='*40}")
            print(f"Test Loss: {test_metrics['loss']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        
        # Close wandb
        if self.config.use_wandb:
            wandb.finish()
        
        return {
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'test_metrics': test_metrics
        }
    
    def predict(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions on new data"""
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            logits = self.model(features)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        return predictions, probs


def train_supervised(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    model: Optional[nn.Module] = None,
    config: Optional[SupervisedConfig] = None,
    device: str = 'cuda'
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
    if model is None:
        model = PolicyNetwork()
    
    if config is None:
        config = SupervisedConfig()
    
    trainer = SupervisedTrainer(model, config, device)
    results = trainer.train(train_loader, val_loader, test_loader)
    
    return model, results
"""
Model utilities: weight initialization, parameter counting, optimization helpers
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json


def initialize_weights(
    model: nn.Module,
    initialization_type: str = 'xavier_uniform',
    gain: float = 0.5
) -> None:
    """
    Initialize model weights with specified method
    
    Args:
        model: PyTorch model
        initialization_type: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 
                             'kaiming_normal', 'orthogonal', 'zeros'
        gain: Gain factor for initialization
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if initialization_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif initialization_type == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif initialization_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif initialization_type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif initialization_type == 'orthogonal':
                nn.init.orthogonal_(module.weight, gain=gain)
            elif initialization_type == 'zeros':
                nn.init.zeros_(module.weight)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    # Count per layer
    per_layer = {}
    for name, param in model.named_parameters():
        per_layer[name] = param.numel()
    
    return {
        'trainable': trainable,
        'total': total,
        'non_trainable': total - trainable,
        'per_layer': per_layer
    }


def freeze_layers(
    model: nn.Module,
    layers_to_freeze: Union[List[str], str] = 'all',
    unfreeze_last_n: int = 0
) -> None:
    """
    Freeze specific layers of the model
    
    Args:
        model: PyTorch model
        layers_to_freeze: List of layer names to freeze, or 'all' for all layers
        unfreeze_last_n: Keep last N layers trainable
    """
    layer_names = list(model.state_dict().keys())
    
    if unfreeze_last_n > 0:
        trainable_layers = set(layer_names[-unfreeze_last_n:])
    else:
        trainable_layers = set()
    
    for name, param in model.named_parameters():
        if layers_to_freeze == 'all':
            if unfreeze_last_n > 0 and name in trainable_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif name in layers_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_layers(model: nn.Module) -> None:
    """Unfreeze all layers of the model"""
    for param in model.parameters():
        param.requires_grad = True


def get_activation_function(name: str) -> nn.Module:
    """
    Get activation function by name
    
    Args:
        name: Activation function name
        
    Returns:
        Activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'relu6': nn.ReLU6(),
        'leaky_relu': nn.LeakyReLU(0.01),
        'leaky_relu_02': nn.LeakyReLU(0.2),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'softplus': nn.Softplus(),
        'softsign': nn.Softsign(),
        'none': nn.Identity()
    }
    return activations.get(name, nn.ReLU())


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with specified parameters
    
    Args:
        model: PyTorch model
        optimizer_name: 'adam', 'sgd', 'adamw', 'rmsprop', 'adagrad'
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
        'adagrad': torch.optim.Adagrad,
        'nadam': torch.optim.NAdam,
        'radam': torch.optim.RAdam
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    optimizer_class = optimizers[optimizer_name.lower()]
    
    if optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', False)
        return optimizer_class(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    return optimizer_class(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        **{k: v for k, v in kwargs.items() if k != 'momentum'}
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'step',
    step_size: int = 30,
    gamma: float = 0.1,
    patience: int = 10,
    factor: float = 0.5,
    min_lr: float = 1e-7,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: 'step', 'cosine', 'exponential', 'reduce_on_plateau', 'cyclic'
        step_size: Step size for step scheduler
        gamma: Decay factor
        patience: Patience for reduce_on_plateau
        factor: Factor for reduce_on_plateau
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Learning rate scheduler
    """
    schedulers = {
        'step': torch.optim.lr_scheduler.StepLR,
        'multi_step': torch.optim.lr_scheduler.MultiStepLR,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'cyclic': torch.optim.lr_scheduler.CyclicLR
    }
    
    if scheduler_name.lower() not in schedulers:
        return None
    
    if scheduler_name.lower() == 'step':
        return schedulers['step'](optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'multi_step':
        milestones = kwargs.get('milestones', [30, 60, 90])
        return schedulers['multi_step'](optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 100)
        return schedulers['cosine'](optimizer, T_max=T_max, eta_min=min_lr)
    elif scheduler_name.lower() == 'exponential':
        return schedulers['exponential'](optimizer, gamma=gamma)
    elif scheduler_name.lower() == 'plateau':
        return schedulers['plateau'](optimizer, patience=patience, factor=factor, min_lr=min_lr)
    elif scheduler_name.lower() == 'cyclic':
        base_lr = kwargs.get('base_lr', 1e-5)
        max_lr = kwargs.get('max_lr')
        return schedulers['cyclic'](optimizer, base_lr=base_lr, max_lr=max_lr)
    
    return None


class ModelCheckpoint:
    """
    Model checkpoint saver and loader
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'experiments/checkpoints',
        save_best_only: bool = True,
        monitor: str = 'val_loss',
        mode: str = 'min',
        max_keep: int = 5
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save when metric improves
            monitor: Metric to monitor
            mode: 'min' or 'max' for metric
            max_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.max_keep = max_keep
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_history = []
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict] = None,
        name: str = 'checkpoint'
    ) -> bool:
        """
        Save model checkpoint
        
        Returns:
            True if saved, False if skipped
        """
        if metrics is None:
            metrics = {}
        
        current_value = metrics.get(self.monitor)
        
        if self.save_best_only and current_value is not None:
            if self.mode == 'min' and current_value >= self.best_value:
                return False
            if self.mode == 'max' and current_value <= self.best_value:
                return False
            
            self.best_value = current_value
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save file
        if self.save_best_only and current_value is not None:
            filename = f"{name}_best_{self.monitor}={current_value:.4f}.pt"
        else:
            filename = f"{name}_epoch_{epoch}.pt"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # Manage history
        self.checkpoint_history.append(filepath)
        if len(self.checkpoint_history) > self.max_keep:
            oldest = self.checkpoint_history.pop(0)
            if oldest.exists():
                oldest.unlink()
        
        return True
    
    def load(
        self,
        model: nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {})
        }


class EarlyStopping:
    """
    Early stopping handler
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to be considered improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
    
    def update(self, value: float) -> bool:
        """
        Update early stopping state
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.should_stop = False


class GradientClipping:
    """
    Gradient clipping utilities
    """
    
    @staticmethod
    def clip_norm(model: nn.Module, max_norm: float = 1.0) -> float:
        """Clip gradients by norm"""
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm.item()
    
    @staticmethod
    def clip_value(model: nn.Module, max_value: float = 1.0) -> None:
        """Clip gradients by value"""
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-max_value, max_value)
    
    @staticmethod
    def get_gradient_norm(model: nn.Module) -> float:
        """Get total gradient norm"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'param_size_mb': param_size / (1024 ** 2),
        'buffer_size_mb': buffer_size / (1024 ** 2),
        'total_mb': total_mb
    }


def print_model_summary(model: nn.Module, input_size: tuple = None, device: str = 'cpu'):
    """Print model summary information"""
    print("\n" + "=" * 60)
    print(f"MODEL SUMMARY: {model.__class__.__name__}")
    print("=" * 60)
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    
    # Model size
    size_info = get_model_size(model)
    print(f"Model size: {size_info['total_mb']:.2f} MB")
    
    # Architecture summary
    print("\n" + "-" * 40)
    print("ARCHITECTURE:")
    print("-" * 40)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            print(f"  {name}: {module.__class__.__name__}")
            if hasattr(module, 'in_features'):
                print(f"    in={module.in_features}, out={module.out_features}")
            elif hasattr(module, 'in_channels'):
                print(f"    in={module.in_channels}, out={module.out_channels}")
    
    # Input/output size test
    if input_size is not None:
        try:
            test_input = torch.randn(1, *input_size).to(device)
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            print(f"\nInput shape: {test_input.shape}")
            print(f"Output shape: {output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}")
        except Exception as e:
            print(f"Could not test forward pass: {e}")
    
    print("=" * 60 + "\n")
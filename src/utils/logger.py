"""
Logging utilities for poker agent
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler


class Logger:
    """
    Custom logger for poker agent with file and console output
    """
    
    def __init__(
        self,
        name: str = "poker_agent",
        log_dir: str = "experiments/logs",
        log_level: str = "INFO",
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Separate error log
        error_log_file = self.log_dir / f"{name}_error.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """Log exception with traceback"""
        self.logger.exception(message)
    
    def log_hand_start(self, hand_id: str, hand_number: int):
        """Log hand start"""
        self.info(f"Hand {hand_number} started: {hand_id}")
    
    def log_hand_end(self, hand_id: str, winner: str, pot: float):
        """Log hand end"""
        self.info(f"Hand {hand_id} ended - Winner: {winner}, Pot: ${pot:.2f}")
    
    def log_action(self, player: str, action: str, amount: float, stack: float):
        """Log player action"""
        self.debug(f"{player}: {action} ${amount:.2f} (stack: ${stack:.2f})")
    
    def log_training_progress(self, epoch: int, metrics: dict):
        """Log training progress"""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
        self.info(f"Epoch {epoch}: {metrics_str}")
    
    def log_evaluation(self, agent_name: str, results: dict):
        """Log evaluation results"""
        self.info(f"Evaluation of {agent_name}:")
        for metric, value in results.items():
            if isinstance(value, float):
                self.info(f"  {metric}: {value:.4f}")
            else:
                self.info(f"  {metric}: {value}")


# Global logger instance
_default_logger: Optional[Logger] = None


def setup_logger(
    name: str = "poker_agent",
    log_dir: str = "experiments/logs",
    log_level: str = "INFO"
) -> Logger:
    """
    Setup global logger
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    global _default_logger
    _default_logger = Logger(name, log_dir, log_level)
    return _default_logger


def get_logger() -> Logger:
    """
    Get global logger instance
    
    Returns:
        Logger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger


class TrainingLogger:
    """
    Specialized logger for training metrics
    """
    
    def __init__(self, log_dir: str = "experiments/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "training_metrics.csv"
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        """Initialize metrics CSV file with headers"""
        if not self.metrics_file.exists():
            import csv
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    
    def log_metrics(self, epoch: int, metrics: dict):
        """Log training metrics to CSV"""
        import csv
        from datetime import datetime
        
        row = [
            datetime.now().isoformat(),
            epoch,
            metrics.get('train_loss', 0),
            metrics.get('train_acc', 0),
            metrics.get('val_loss', 0),
            metrics.get('val_acc', 0),
            metrics.get('lr', 0)
        ]
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
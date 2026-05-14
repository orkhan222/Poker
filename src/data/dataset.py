"""
PyTorch Dataset class for poker data
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List

# Relative import əvəzinə absolute import istifadə edin
try:
    from src.data.features import PokerFeatureExtractor, PokerState
except ImportError:
    from features import PokerFeatureExtractor, PokerState


class PokerDataset(Dataset):
    """
    PyTorch Dataset for poker hand data
    
    Each sample: (features, action_label)
    """
    
    def __init__(
        self,
        players_df: pd.DataFrame,
        actions_df: pd.DataFrame,
        feature_extractor: Optional[PokerFeatureExtractor] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset
        
        Args:
            players_df: Players DataFrame with hole cards and stack info
            actions_df: Actions DataFrame with actions taken
            feature_extractor: Feature extractor instance
            max_samples: Maximum number of samples (for debugging)
        """
        self.players_df = players_df
        self.actions_df = actions_df
        self.feature_extractor = feature_extractor or PokerFeatureExtractor()
        
        # Merge data
        self.data = self._merge_data()
        
        if max_samples:
            self.data = self.data.head(max_samples)
        
        print(f"📊 Dataset created with {len(self.data)} samples")
    
    def _merge_data(self) -> pd.DataFrame:
        """Merge players and actions data"""
        # Check if action_encoded exists, if not create it
        if 'action_encoded' not in self.actions_df.columns:
            # Create action encoding
            action_map = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}
            if 'action' in self.actions_df.columns:
                self.actions_df['action_encoded'] = self.actions_df['action'].map(action_map).fillna(0).astype(int)
            else:
                self.actions_df['action_encoded'] = 0
        
        # Merge on hand_id and position
        if 'position' in self.players_df.columns and 'position' in self.actions_df.columns:
            merged = self.players_df.merge(
                self.actions_df,
                on=['hand_id', 'position'],
                how='inner'
            )
        else:
            merged = self.players_df.merge(
                self.actions_df,
                on=['hand_id'],
                how='inner'
            )
        
        # Drop rows with missing critical values
        if 'action_encoded' in merged.columns:
            merged = merged.dropna(subset=['action_encoded'])
        
        return merged
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Returns:
            Tuple of (features_tensor, action_label)
        """
        row = self.data.iloc[idx]
        
        # Create state from row data
        hole_cards = row.get('cards', '').split() if pd.notna(row.get('cards')) else []
        board_cards = row.get('board_cards', '').split() if pd.notna(row.get('board_cards')) else []
        
        # Handle NaN values
        agent_stack = row.get('starting_stack', 1000)
        if pd.isna(agent_stack):
            agent_stack = 1000
        
        pot = row.get('pot_from_stacks', 0)
        if pd.isna(pot):
            pot = 0
        
        street = row.get('street', 'preflop')
        if pd.isna(street):
            street = 'preflop'
        
        # Legal actions (simplified - in production you'd compute from game state)
        legal_actions = ['fold', 'check', 'call', 'raise']
        
        state = PokerState(
            hole_cards=hole_cards,
            board_cards=board_cards,
            agent_stack=float(agent_stack),
            pot=float(pot),
            current_bet=0.0,
            street=street,
            legal_actions=legal_actions
        )
        
        # Extract features
        features = self.feature_extractor.state_to_tensor(state)
        
        # Get action label
        action_label = row.get('action_encoded', 0)
        if pd.isna(action_label):
            action_label = 0
        
        return features, torch.tensor(action_label, dtype=torch.long)
    
    def get_action_distribution(self) -> Dict[int, int]:
        """Get distribution of actions in dataset"""
        if 'action_encoded' not in self.data.columns:
            return {}
        
        distribution = self.data['action_encoded'].value_counts().to_dict()
        return distribution


def create_dataloaders(
    players_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        players_df: Players DataFrame
        actions_df: Actions DataFrame
        batch_size: Batch size for dataloaders
        train_split: Training set proportion
        val_split: Validation set proportion
        test_split: Test set proportion
        random_seed: Random seed for reproducibility
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = PokerDataset(players_df, actions_df)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    torch.manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n📊 Dataloaders created:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def create_sample_dataset(num_samples: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a sample dataset for testing
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (players_df, actions_df)
    """
    np.random.seed(42)
    
    # Sample players data
    players_data = []
    actions_data = []
    
    actions_list = ['fold', 'call', 'raise', 'bet', 'check']
    action_map = {'fold': 0, 'call': 1, 'raise': 2, 'bet': 3, 'check': 4}
    
    streets = ['preflop', 'flop', 'turn', 'river']
    positions = ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN']
    card_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    card_suits = ['h', 'd', 'c', 's']
    
    for i in range(num_samples):
        hand_id = f"hand_{i:05d}"
        
        # Generate random hole cards
        rank1, rank2 = np.random.choice(card_ranks, 2)
        suit1, suit2 = np.random.choice(card_suits, 2)
        cards = f"{rank1}{suit1} {rank2}{suit2}"
        
        # Random values
        starting_stack = np.random.uniform(500, 5000)
        ending_stack = starting_stack + np.random.uniform(-500, 500)
        stack_delta = ending_stack - starting_stack
        
        # Random board cards
        num_board = np.random.choice([0, 3, 4, 5])
        board_cards = []
        for _ in range(num_board):
            rank = np.random.choice(card_ranks)
            suit = np.random.choice(card_suits)
            board_cards.append(f"{rank}{suit}")
        
        position = np.random.choice(positions)
        
        players_data.append({
            'hand_id': hand_id,
            'position': position,
            'cards': cards,
            'starting_stack': starting_stack,
            'ending_stack': ending_stack,
            'stack_delta': stack_delta,
            'board_cards': ' '.join(board_cards),
            'pot_from_stacks': np.random.uniform(0, 2000),
            'street': np.random.choice(streets)
        })
        
        # Random actions (multiple per hand possible)
        num_actions = np.random.randint(1, 5)
        for j in range(num_actions):
            action = np.random.choice(actions_list)
            actions_data.append({
                'hand_id': hand_id,
                'position': position,
                'action': action,
                'action_encoded': action_map.get(action, 0),
                'street': np.random.choice(streets),
                'frame_id': j
            })
    
    players_df = pd.DataFrame(players_data)
    actions_df = pd.DataFrame(actions_data)
    
    print(f"✅ Sample dataset created: {len(players_df)} players, {len(actions_df)} actions")
    print(f"   Action distribution: {actions_df['action'].value_counts().to_dict()}")
    
    return players_df, actions_df
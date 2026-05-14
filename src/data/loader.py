"""
Data loader module - Load CSV files and merge them
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_hands(data_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Load hands.csv file
    
    Args:
        data_dir: Directory containing processed CSV files
        
    Returns:
        DataFrame with hand information
    """
    file_path = Path(data_dir) / 'hands.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Hands file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} hands from {file_path}")
    return df


def load_players(data_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Load players.csv file
    
    Args:
        data_dir: Directory containing processed CSV files
        
    Returns:
        DataFrame with player information
    """
    file_path = Path(data_dir) / 'players.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Players file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} player records from {file_path}")
    return df


def load_actions(data_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Load actions.csv file
    
    Args:
        data_dir: Directory containing processed CSV files
        
    Returns:
        DataFrame with action information
    """
    file_path = Path(data_dir) / 'actions.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Actions file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} actions from {file_path}")
    return df


def load_stack_events(data_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Load stack_events.csv file
    
    Args:
        data_dir: Directory containing processed CSV files
        
    Returns:
        DataFrame with stack event information
    """
    file_path = Path(data_dir) / 'stack_events.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Stack events file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} stack events from {file_path}")
    return df


def load_all_data(data_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all CSV files
    
    Args:
        data_dir: Directory containing processed CSV files
        
    Returns:
        Tuple of (hands, players, actions, stack_events) DataFrames
    """
    hands = load_hands(data_dir)
    players = load_players(data_dir)
    actions = load_actions(data_dir)
    stack_events = load_stack_events(data_dir)
    
    print(f"\n📊 Data Summary:")
    print(f"   Hands: {len(hands)}")
    print(f"   Players: {len(players)}")
    print(f"   Actions: {len(actions)}")
    print(f"   Stack Events: {len(stack_events)}")
    
    return hands, players, actions, stack_events


def merge_all_data(
    hands: pd.DataFrame,
    players: pd.DataFrame,
    actions: pd.DataFrame,
    stack_events: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge all data into a single DataFrame for analysis
    
    Args:
        hands: Hands DataFrame
        players: Players DataFrame
        actions: Actions DataFrame
        stack_events: Optional Stack events DataFrame
        
    Returns:
        Merged DataFrame
    """
    # Merge players with hands
    merged = players.merge(hands, on='hand_id', how='left', suffixes=('', '_hand'))
    
    # Merge actions if needed (for sequence data)
    if actions is not None and len(actions) > 0:
        # For each hand, we might want to merge actions
        pass
    
    print(f"✅ Merged data: {len(merged)} rows")
    return merged


def get_hand_sequence(hand_id: str, actions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get all actions for a specific hand in chronological order
    
    Args:
        hand_id: Unique hand identifier
        actions_df: Actions DataFrame
        
    Returns:
        DataFrame with actions sorted by frame_id
    """
    hand_actions = actions_df[actions_df['hand_id'] == hand_id].copy()
    hand_actions = hand_actions.sort_values('frame_id')
    return hand_actions


def get_player_history(
    hand_id: str,
    player_position: str,
    actions_df: pd.DataFrame,
    stack_events_df: pd.DataFrame
) -> Dict:
    """
    Get complete history for a specific player in a hand
    
    Args:
        hand_id: Unique hand identifier
        player_position: Player position (e.g., 'SB', 'BB')
        actions_df: Actions DataFrame
        stack_events_df: Stack events DataFrame
        
    Returns:
        Dictionary with player's actions and stack changes
    """
    player_actions = actions_df[
        (actions_df['hand_id'] == hand_id) & 
        (actions_df['player_position'] == player_position)
    ].sort_values('frame_id')
    
    player_stacks = stack_events_df[
        (stack_events_df['hand_id'] == hand_id) & 
        (stack_events_df['player_position'] == player_position)
    ].sort_values('frame_id')
    
    return {
        'hand_id': hand_id,
        'player_position': player_position,
        'actions': player_actions.to_dict('records'),
        'stack_events': player_stacks.to_dict('records'),
        'num_actions': len(player_actions),
        'num_stack_changes': len(player_stacks)
    }
"""
Data preprocessing module - Cleaning, normalization, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Main preprocessor class for poker data"""
    
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.position_encoder = LabelEncoder()
        self.street_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, actions_df: pd.DataFrame, players_df: pd.DataFrame):
        """
        Fit encoders on data
        
        Args:
            actions_df: Actions DataFrame
            players_df: Players DataFrame
        """
        # Fit action encoder
        if 'action' in actions_df.columns:
            self.action_encoder.fit(actions_df['action'].dropna())
        
        # Fit position encoder
        if 'position' in players_df.columns:
            self.position_encoder.fit(players_df['position'].dropna())
        
        # Fit street encoder
        if 'street' in actions_df.columns:
            self.street_encoder.fit(actions_df['street'].dropna())
        
        self.is_fitted = True
        print("✅ Preprocessor fitted")
        
    def preprocess(
        self,
        hands_df: pd.DataFrame,
        players_df: pd.DataFrame,
        actions_df: pd.DataFrame,
        stack_events_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Preprocess all DataFrames
        
        Args:
            hands_df: Hands DataFrame
            players_df: Players DataFrame
            actions_df: Actions DataFrame
            stack_events_df: Optional Stack events DataFrame
            
        Returns:
            Preprocessed DataFrames
        """
        # Process hands
        hands_df = self._preprocess_hands(hands_df)
        
        # Process players
        players_df = self._preprocess_players(players_df)
        
        # Process actions
        actions_df = self._preprocess_actions(actions_df)
        
        # Process stack events
        if stack_events_df is not None:
            stack_events_df = self._preprocess_stack_events(stack_events_df)
        
        print("✅ Data preprocessing complete")
        return hands_df, players_df, actions_df, stack_events_df
    
    def _preprocess_hands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess hands DataFrame"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['hand_id'])
        
        # Handle missing values
        df['board_cards'] = df['board_cards'].fillna('')
        df['pot_from_stacks'] = df['pot_from_stacks'].fillna(0)
        df['pot_from_recognition'] = df['pot_from_recognition'].fillna(0)
        
        # Create average pot
        df['pot_avg'] = (df['pot_from_stacks'] + df['pot_from_recognition']) / 2
        
        # Count cards on board
        df['num_board_cards'] = df['board_cards'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        # Create hand complexity score
        df['hand_complexity'] = df['total_actions'] / df['num_board_cards'].clip(lower=1)
        
        return df
    
    def _preprocess_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess players DataFrame"""
        df = df.copy()
        
        # Handle missing values
        df['nickname'] = df['nickname'].fillna('unknown')
        df['cards'] = df['cards'].fillna('')
        df['starting_stack'] = df['starting_stack'].fillna(1000)
        df['ending_stack'] = df['ending_stack'].fillna(1000)
        df['stack_delta'] = df['stack_delta'].fillna(0)
        
        # Normalize stacks
        df['starting_stack_norm'] = df['starting_stack'] / 1000
        df['ending_stack_norm'] = df['ending_stack'] / 1000
        df['stack_delta_norm'] = df['stack_delta'] / 1000
        
        # Encode positions if fitted
        if self.is_fitted and 'position' in df.columns:
            valid_positions = df['position'].dropna()
            known_positions = [p for p in valid_positions if p in self.position_encoder.classes_]
            if known_positions:
                df['position_encoded'] = self.position_encoder.transform(known_positions)
            else:
                df['position_encoded'] = 0
        else:
            df['position_encoded'] = 0
        
        # Hand strength feature
        df['hand_strength'] = df['cards'].apply(self._calculate_hand_strength)
        
        # Has pocket pair
        df['has_pocket_pair'] = df['cards'].apply(self._has_pocket_pair)
        
        # Has suited cards
        df['has_suited_cards'] = df['cards'].apply(self._has_suited_cards)
        
        # Has high cards
        df['has_high_cards'] = df['cards'].apply(self._has_high_cards)
        
        return df
    
    def _preprocess_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess actions DataFrame"""
        df = df.copy()
        
        # Handle missing values
        df['action'] = df['action'].fillna('fold')
        df['street'] = df['street'].fillna('preflop')
        df['frame_id'] = df['frame_id'].fillna(0)
        
        # Encode actions if fitted
        if self.is_fitted and 'action' in df.columns:
            valid_actions = df['action'].dropna()
            known_actions = [a for a in valid_actions if a in self.action_encoder.classes_]
            if known_actions:
                df['action_encoded'] = self.action_encoder.transform(known_actions)
            else:
                df['action_encoded'] = 0
        else:
            df['action_encoded'] = 0
        
        # Encode streets if fitted
        if self.is_fitted and 'street' in df.columns:
            valid_streets = df['street'].dropna()
            known_streets = [s for s in valid_streets if s in self.street_encoder.classes_]
            if known_streets:
                df['street_encoded'] = self.street_encoder.transform(known_streets)
            else:
                df['street_encoded'] = 0
        else:
            df['street_encoded'] = 0
        
        # Create action type categories
        df['is_aggressive'] = df['action'].apply(lambda x: x in ['bet', 'raise', 'all_in'])
        df['is_passive'] = df['action'].apply(lambda x: x in ['call', 'check'])
        df['is_fold'] = df['action'] == 'fold'
        
        return df
    
    def _preprocess_stack_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess stack events DataFrame"""
        df = df.copy()
        
        # Handle missing values
        df['stack'] = df['stack'].fillna(1000)
        df['diff'] = df['diff'].fillna(0)
        
        # Normalize stack values
        df['stack_norm'] = df['stack'] / 1000
        df['diff_norm'] = df['diff'] / 1000
        
        return df
    
    @staticmethod
    def _calculate_hand_strength(cards: str) -> float:
        """Calculate hand strength based on hole cards"""
        if not cards or pd.isna(cards):
            return 0.0
        
        card_list = cards.split()
        if len(card_list) < 2:
            return 0.0
        
        ranks = []
        suits = []
        
        rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        
        for card in card_list[:2]:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1] if len(card) == 2 else card[2]
                
                if rank in rank_map:
                    ranks.append(rank_map[rank])
                elif rank.isdigit():
                    ranks.append(int(rank))
                suits.append(suit)
        
        if len(ranks) < 2:
            return 0.0
        
        # Check for pocket pair
        is_pair = ranks[0] == ranks[1]
        
        # Check for suited
        is_suited = suits[0] == suits[1] if len(suits) >= 2 else False
        
        # Calculate strength (max ~1.0)
        strength = 0.0
        
        if is_pair:
            strength += 0.5
            # High pair bonus
            if ranks[0] >= 12:
                strength += 0.3
            elif ranks[0] >= 10:
                strength += 0.15
        
        if is_suited:
            strength += 0.2
        
        # High card bonus
        max_rank = max(ranks)
        strength += (max_rank - 2) / 12 * 0.3
        
        return min(strength, 1.0)
    
    @staticmethod
    def _has_pocket_pair(cards: str) -> bool:
        """Check if player has a pocket pair"""
        if not cards or pd.isna(cards):
            return False
        
        card_list = cards.split()
        if len(card_list) < 2:
            return False
        
        ranks = []
        for card in card_list[:2]:
            if len(card) >= 2:
                ranks.append(card[0])
        
        return len(ranks) >= 2 and ranks[0] == ranks[1] if ranks else False
    
    @staticmethod
    def _has_suited_cards(cards: str) -> bool:
        """Check if hole cards are suited"""
        if not cards or pd.isna(cards):
            return False
        
        card_list = cards.split()
        if len(card_list) < 2:
            return False
        
        suits = []
        for card in card_list[:2]:
            if len(card) >= 2:
                suit = card[1] if len(card) == 2 else card[2]
                suits.append(suit)
        
        return len(suits) >= 2 and suits[0] == suits[1] if suits else False
    
    @staticmethod
    def _has_high_cards(cards: str) -> bool:
        """Check if player has high cards (A, K, Q, J)"""
        if not cards or pd.isna(cards):
            return False
        
        high_ranks = {'A', 'K', 'Q', 'J'}
        card_list = cards.split()
        
        for card in card_list[:2]:
            if len(card) >= 2 and card[0] in high_ranks:
                return True
        
        return False


def clean_data(
    hands_df: pd.DataFrame,
    players_df: pd.DataFrame,
    actions_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Quick data cleaning without fitting
    
    Args:
        hands_df: Hands DataFrame
        players_df: Players DataFrame
        actions_df: Actions DataFrame
        
    Returns:
        Cleaned DataFrames
    """
    # Remove rows with missing critical data
    players_df = players_df.dropna(subset=['position', 'starting_stack'])
    actions_df = actions_df.dropna(subset=['action'])
    
    # Remove duplicates
    hands_df = hands_df.drop_duplicates(subset=['hand_id'])
    
    # Clip extreme values
    if 'stack_delta' in players_df.columns:
        players_df['stack_delta'] = players_df['stack_delta'].clip(-5000, 5000)
    
    return hands_df, players_df, actions_df


def normalize_features(
    df: pd.DataFrame,
    columns: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize specified columns
    
    Args:
        df: DataFrame to normalize
        columns: List of column names to normalize
        scaler: Optional pre-fitted scaler
        
    Returns:
        Normalized DataFrame and scaler
    """
    df = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    
    return df, scaler
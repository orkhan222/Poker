"""
Gymnasium-compatible poker environment for RL training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

from .game_state import GameState, PlayerState, Street, Action
from .rules import PokerRules, BlindStructure, BettingRules
from .hand_ranker import HandEvaluator


@dataclass
class PokerEnvConfig:
    """Configuration for poker environment"""
    num_players: int = 6
    starting_stack: float = 1000.0
    small_blind: float = 5.0
    big_blind: float = 10.0
    max_steps_per_hand: int = 200
    reward_scaling: float = 0.01
    agent_position: int = 0  # Position of the learning agent


class PokerEnv(gym.Env):
    """
    Texas Hold'em Poker Environment for RL
    
    Observation space: Feature vector
    Action space: 6 discrete actions (fold, check, call, bet, raise, all-in)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, config: PokerEnvConfig = None):
        super().__init__()
        
        self.config = config or PokerEnvConfig()
        self.num_players = self.config.num_players
        self.starting_stack = self.config.starting_stack
        
        # Action space: 6 actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: feature vector
        # 52 (hole) + 52 (board) + 1 (stack) + 1 (pot) + 1 (current bet) + 
        # 4 (street one-hot) + 6 (legal actions) + 10 (history) + 10 (opponent info)
        obs_dim = 52 + 52 + 1 + 1 + 1 + 4 + 6 + 10 + 10
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize game components
        self.blind_structure = BlindStructure(
            self.config.small_blind, 
            self.config.big_blind
        )
        self.betting_rules = BettingRules(min_bet=self.config.big_blind)
        self.poker_rules = PokerRules()
        
        # State
        self.state: Optional[GameState] = None
        self.steps_taken = 0
        self.hand_reward = 0.0
        
        # Rendering
        self.render_mode = None
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new hand"""
        super().reset(seed=seed)
        
        # Create new deck and shuffle
        deck = self.poker_rules.create_deck()
        if seed is not None:
            deck = self.poker_rules.shuffle_deck(deck, seed)
        
        # Initialize players
        players = []
        for i in range(self.num_players):
            players.append(PlayerState(
                position=i,
                stack=self.starting_stack,
                nickname=f"Player_{i}" if i != self.config.agent_position else "Agent"
            ))
        
        # Deal hole cards
        for player in players:
            player.hole_cards, deck = self.poker_rules.deal_cards(deck, 2)
        
        # Initialize game state
        self.state = GameState(
            hand_id=f"hand_{np.random.randint(10000)}",
            hand_number=0,
            players=players,
            dealer_idx=0,
            board_cards=[],
            street=Street.PREFLOP
        )
        
        # Post blinds
        self.poker_rules.post_blinds(self.state, self.blind_structure)
        
        # Set current player (UTG after blinds)
        self.state.current_player_idx = (self.state.dealer_idx + 3) % self.num_players
        
        # Reset tracking
        self.steps_taken = 0
        self.hand_reward = 0.0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take an action in the environment
        
        Args:
            action: Integer action (0-5) corresponding to Action enum
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.steps_taken += 1
        
        # Convert action to enum
        action_enum = Action(action)
        player = self.state.current_player
        
        # Check if action is legal
        legal_actions = self.state.get_legal_actions()
        if action_enum not in legal_actions:
            # Illegal action -> penalize and fold
            reward = -20.0
            action_enum = Action.FOLD
        
        # Calculate bet amount
        bet_amount = self.state.get_bet_amount(action_enum)
        
        # Apply action
        amount_paid = self.state.apply_action(action_enum, bet_amount)
        
        # Calculate immediate reward
        reward = self._calculate_reward(action_enum, amount_paid)
        self.hand_reward += reward
        
        # Advance game
        self._advance_game()
        
        # Check if hand is over
        terminated = self.state.is_terminal
        truncated = self.steps_taken >= self.config.max_steps_per_hand
        
        # Final reward at hand end
        if terminated:
            reward += self._get_final_reward()
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _advance_game(self):
        """Advance the game to next player or next street"""
        
        # Move to next player
        next_idx = self.state.advance_to_next_player()
        
        if next_idx == -1:
            # No more players to act, check if round is complete
            if self.state.is_betting_round_complete():
                if self.state.street == Street.RIVER:
                    # Hand is over, go to showdown
                    self._resolve_showdown()
                else:
                    # Move to next street
                    self.state.next_street()
                    
                    # Deal community cards
                    deck = self.poker_rules.create_deck()  # In real impl, track deck
                    if self.state.street == Street.FLOP:
                        self.state.board_cards, _ = self.poker_rules.deal_flop(deck)
                    elif self.state.street == Street.TURN:
                        turn_card, _ = self.poker_rules.deal_turn(deck)
                        self.state.board_cards.append(turn_card)
                    elif self.state.street == Street.RIVER:
                        river_card, _ = self.poker_rules.deal_river(deck)
                        self.state.board_cards.append(river_card)
                    
                    # Reset for new round
                    for player in self.state.players:
                        player.current_bet = 0
                        player.has_acted = False
    
    def _resolve_showdown(self):
        """Determine winner and distribute pot"""
        if self.state.num_active_players == 1:
            # Only one player left
            winner_idx = self._get_last_active_player()
            self.state.winners = [winner_idx]
            self.state.winner_amounts = [self.state.pot]
            self.state.players[winner_idx].stack += self.state.pot
        else:
            # Multiple players - compare hands
            best_hand = None
            winners = []
            
            for idx, player in enumerate(self.state.players):
                if player.is_active and not player.is_all_in:
                    hand_strength = HandEvaluator.evaluate_hand(
                        player.hole_cards, 
                        self.state.board_cards
                    )
                    
                    if best_hand is None or hand_strength > best_hand:
                        best_hand = hand_strength
                        winners = [idx]
                    elif hand_strength == best_hand:
                        winners.append(idx)
            
            # Split pot among winners
            pot_share = self.state.pot / len(winners)
            for winner_idx in winners:
                self.state.players[winner_idx].stack += pot_share
                self.state.winners.append(winner_idx)
                self.state.winner_amounts.append(pot_share)
        
        self.state.is_terminal = True
    
    def _get_last_active_player(self) -> int:
        """Get index of the last active player"""
        for idx, player in enumerate(self.state.players):
            if player.is_active:
                return idx
        return -1
    
    def _calculate_reward(self, action: Action, amount_paid: float) -> float:
        """Calculate immediate reward for an action"""
        reward = 0.0
        
        # Negative reward for paying chips
        reward -= amount_paid * self.config.reward_scaling
        
        # Small reward for winning pot (will be added at end)
        # Small penalty for folding when pot is small
        if action == Action.FOLD and self.state.pot < 100:
            reward -= 5.0 * self.config.reward_scaling
        
        return reward
    
    def _get_final_reward(self) -> float:
        """Get final reward at end of hand"""
        agent = self.state.players[self.config.agent_position]
        final_stack = agent.stack
        net_profit = final_stack - self.starting_stack
        
        # Scale reward
        reward = net_profit * self.config.reward_scaling
        
        # Bonus for winning
        if self.config.agent_position in self.state.winners:
            reward += 10.0 * self.config.reward_scaling
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Convert game state to observation vector"""
        obs = []
        
        # Agent's hole cards (52-dim one-hot)
        agent = self.state.players[self.config.agent_position]
        hole_one_hot = self._cards_to_one_hot(agent.hole_cards)
        obs.extend(hole_one_hot)
        
        # Board cards (52-dim one-hot)
        board_one_hot = self._cards_to_one_hot(self.state.board_cards)
        obs.extend(board_one_hot)
        
        # Agent stack (normalized)
        obs.append(min(agent.stack / 10000, 1.0))
        
        # Pot size (normalized)
        obs.append(min(self.state.pot / 20000, 1.0))
        
        # Current bet to call (normalized)
        call_amount = self.state.current_bet - agent.current_bet
        obs.append(min(call_amount / 5000, 1.0))
        
        # Street (one-hot)
        street_one_hot = [0.0, 0.0, 0.0, 0.0]
        street_one_hot[self.state.street.value] = 1.0
        obs.extend(street_one_hot)
        
        # Legal actions (6-dim binary)
        legal_actions = self.state.get_legal_actions()
        legal_vec = [1.0 if Action(i) in legal_actions else 0.0 for i in range(6)]
        obs.extend(legal_vec)
        
        # Opponent information
        for player in self.state.players:
            if player.position != self.config.agent_position:
                obs.append(min(player.stack / 10000, 1.0))
                obs.append(min(player.current_bet / 5000, 1.0))
        
        # Pad to fixed size
        target_size = self.observation_space.shape[0]
        while len(obs) < target_size:
            obs.append(0.0)
        
        return np.array(obs[:target_size], dtype=np.float32)
    
    def _cards_to_one_hot(self, cards: List[str]) -> List[float]:
        """Convert cards to 52-dim one-hot vector"""
        one_hot = [0.0] * 52
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,
                    '9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        suit_map = {'h':0,'d':1,'c':2,'s':3}
        
        for card in cards:
            if len(card) >= 2:
                rank = card[0]
                suit = card[1] if len(card) == 2 else card[2]
                if rank in rank_map and suit in suit_map:
                    idx = rank_map[rank] * 4 + suit_map[suit]
                    if idx < 52:
                        one_hot[idx] = 1.0
        return one_hot
    
    def _get_info(self) -> Dict:
        """Get additional info about current state"""
        agent = self.state.players[self.config.agent_position]
        
        return {
            'hand_id': self.state.hand_id,
            'street': self.state.street.name,
            'pot': self.state.pot,
            'agent_stack': agent.stack,
            'agent_current_bet': agent.current_bet,
            'current_bet': self.state.current_bet,
            'num_active_players': self.state.num_active_players,
            'legal_actions': [a.name for a in self.state.get_legal_actions()],
            'hand_reward_so_far': self.hand_reward
        }
    
    def render(self):
        """Render the current state (simplified)"""
        if self.state is None:
            print("Environment not initialized")
            return
        
        print(f"\n{'='*50}")
        print(f"Hand: {self.state.hand_id} | Street: {self.state.street.name}")
        print(f"Pot: ${self.state.pot:.2f} | Current Bet: ${self.state.current_bet:.2f}")
        print(f"Board: {self.state.board_cards}")
        print(f"{'-'*30}")
        
        for player in self.state.players:
            is_agent = player.position == self.config.agent_position
            marker = "🤖 AGENT" if is_agent else f"👤 {player.nickname}"
            active = "ACTIVE" if player.is_active else "FOLDED"
            all_in = " ALL-IN" if player.is_all_in else ""
            print(f"{marker}: ${player.stack:.2f} | Bet: ${player.current_bet:.2f} | {active}{all_in}")
            if is_agent and player.hole_cards:
                print(f"   Cards: {player.hole_cards}")
        
        legal = [a.name for a in self.state.get_legal_actions()]
        print(f"\nLegal actions: {legal}")
        print(f"{'='*50}\n")
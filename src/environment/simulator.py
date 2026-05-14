"""
Multi-agent poker simulator for training and evaluation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .poker_env import PokerEnv, PokerEnvConfig
from .game_state import Action
from src.agents.base_agent import BaseAgent


@dataclass
class TournamentResult:
    """Results from a tournament or series of hands"""
    num_hands: int = 0
    wins: Dict[str, int] = field(default_factory=dict)
    total_profit: Dict[str, float] = field(default_factory=dict)
    final_stacks: Dict[str, float] = field(default_factory=dict)
    action_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    hand_history: List[Dict] = field(default_factory=list)
    
    def add_hand_result(self, hand_result: Dict):
        """Add a single hand result to the tournament"""
        self.num_hands += 1
        self.hand_history.append(hand_result)
        
        # Update wins
        winner = hand_result.get('winner')
        if winner:
            self.wins[winner] = self.wins.get(winner, 0) + 1
        
        # Update profits and stacks
        for player, profit in hand_result.get('profits', {}).items():
            self.total_profit[player] = self.total_profit.get(player, 0) + profit
        
        for player, stack in hand_result.get('final_stacks', {}).items():
            self.final_stacks[player] = stack
        
        # Update action counts
        for player, actions in hand_result.get('actions', {}).items():
            if player not in self.action_counts:
                self.action_counts[player] = defaultdict(int)
            for action in actions:
                self.action_counts[player][action] += 1
    
    def get_win_rate(self, player_name: str) -> float:
        """Get win rate for a player"""
        if self.num_hands == 0:
            return 0.0
        return self.wins.get(player_name, 0) / self.num_hands
    
    def get_average_profit(self, player_name: str) -> float:
        """Get average profit per hand"""
        if self.num_hands == 0:
            return 0.0
        return self.total_profit.get(player_name, 0) / self.num_hands
    
    def summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'num_hands': self.num_hands,
            'wins': dict(self.wins),
            'win_rates': {p: self.get_win_rate(p) for p in self.wins},
            'total_profit': dict(self.total_profit),
            'avg_profit': {p: self.get_average_profit(p) for p in self.total_profit},
            'final_stacks': dict(self.final_stacks)
        }
    
    def print_summary(self):
        """Print tournament summary"""
        print("\n" + "="*60)
        print("TOURNAMENT RESULTS")
        print("="*60)
        print(f"Total Hands: {self.num_hands}")
        print("-"*40)
        
        for player in self.final_stacks.keys():
            win_rate = self.get_win_rate(player)
            avg_profit = self.get_average_profit(player)
            final_stack = self.final_stacks.get(player, 0)
            
            print(f"\n{player}:")
            print(f"  Win Rate: {win_rate:.3f} ({self.wins.get(player, 0)}/{self.num_hands})")
            print(f"  Avg Profit: ${avg_profit:.2f}")
            print(f"  Final Stack: ${final_stack:.2f}")
            print(f"  Total Profit: ${self.total_profit.get(player, 0):.2f}")
        
        print("="*60)


class MultiAgentSimulator:
    """
    Simulator for running multiple poker agents against each other
    """
    
    def __init__(
        self,
        num_players: int = 6,
        starting_stack: float = 1000,
        small_blind: float = 5,
        big_blind: float = 10,
        verbose: bool = False
    ):
        """
        Initialize simulator
        
        Args:
            num_players: Number of players at the table
            starting_stack: Starting stack for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
            verbose: Print debug information
        """
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.verbose = verbose
        
        self.agents: Dict[int, BaseAgent] = {}  # position -> agent
        self.agent_names: Dict[int, str] = {}
        
        self.config = PokerEnvConfig(
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            agent_position=0  # Will be updated per hand
        )
    
    def register_agent(self, agent: BaseAgent, position: int, name: str = None):
        """
        Register an agent to play at a specific position
        
        Args:
            agent: Agent instance
            position: Seat position (0 to num_players-1)
            name: Optional name for the agent
        """
        self.agents[position] = agent
        self.agent_names[position] = name or f"Agent_{position}"
    
    def run_hand(self, hand_number: int = 0) -> Dict[str, Any]:
        """
        Run a single hand with the registered agents
        
        Args:
            hand_number: Hand number for tracking
            
        Returns:
            Dictionary with hand results
        """
        # Rotate dealer position
        dealer_pos = hand_number % self.num_players
        
        # Create environment with agent at different positions
        # For each hand, we need to map agents to positions
        results = {
            'hand_number': hand_number,
            'dealer_position': dealer_pos,
            'actions': [],
            'final_stacks': {},
            'profits': {},
            'winner': None,
            'hand_id': None
        }
        
        # We need to simulate with all agents
        # This is complex - simplified version:
        env = PokerEnv(self.config)
        obs, info = env.reset()
        done = False
        
        hand_actions = []
        
        while not done:
            current_pos = env.state.current_player_idx
            
            # Get agent for this position
            if current_pos in self.agents:
                agent = self.agents[current_pos]
                state_dict = env.state.to_dict()
                
                # Add agent-specific info
                state_dict['agent_position'] = current_pos
                state_dict['agent_stack'] = env.state.players[current_pos].stack
                
                # Get action from agent
                action_str, bet_size = agent.act(state_dict)
                
                # Convert to action enum
                action_map = {
                    'fold': 0, 'check': 1, 'call': 2,
                    'bet': 3, 'raise': 4, 'all_in': 5
                }
                action = action_map.get(action_str.lower(), 0)
                
                hand_actions.append({
                    'player': self.agent_names.get(current_pos, f"Player_{current_pos}"),
                    'position': current_pos,
                    'action': action_str,
                    'bet_size': bet_size
                })
            else:
                # No agent - take random action
                action = env.action_space.sample()
                hand_actions.append({
                    'player': f"Player_{current_pos}",
                    'position': current_pos,
                    'action': Action(action).name,
                    'bet_size': 0
                })
            
            obs, reward, done, truncated, info = env.step(action)
        
        # Collect results
        for i, player in enumerate(env.state.players):
            name = self.agent_names.get(i, f"Player_{i}")
            results['final_stacks'][name] = player.stack
            results['profits'][name] = player.stack - self.starting_stack
            
            if i in env.state.winners:
                results['winner'] = name
        
        results['hand_id'] = env.state.hand_id
        results['actions'] = hand_actions
        
        if self.verbose:
            print(f"Hand {hand_number}: Winner = {results['winner']}, Pot = ${env.state.pot:.2f}")
        
        return results
    
    def run_tournament(
        self,
        num_hands: int = 100,
        starting_stacks: Optional[Dict[int, float]] = None,
        verbose: bool = False
    ) -> TournamentResult:
        """
        Run a tournament of multiple hands
        
        Args:
            num_hands: Number of hands to play
            starting_stacks: Optional custom starting stacks per position
            verbose: Print progress
            
        Returns:
            TournamentResult object with statistics
        """
        result = TournamentResult()
        
        # Track stacks across hands
        stacks = {pos: self.starting_stack for pos in self.agents.keys()}
        if starting_stacks:
            stacks.update(starting_stacks)
        
        for hand_num in range(num_hands):
            # Update blind structure based on hand number
            blind_level = 1 + (hand_num // 50)  # Increase blinds every 50 hands
            sb = self.small_blind * blind_level
            bb = self.big_blind * blind_level
            
            self.config.small_blind = sb
            self.config.big_blind = bb
            
            # Run hand
            hand_result = self.run_hand(hand_num)
            
            # Update stacks
            for pos, name in self.agent_names.items():
                if name in hand_result['final_stacks']:
                    stacks[pos] = hand_result['final_stacks'][name]
            
            # Format for TournamentResult
            formatted_result = {
                'hand_number': hand_num,
                'winner': hand_result['winner'],
                'final_stacks': hand_result['final_stacks'],
                'profits': hand_result['profits'],
                'actions': {}
            }
            
            # Group actions by player
            for action in hand_result['actions']:
                player = action['player']
                if player not in formatted_result['actions']:
                    formatted_result['actions'][player] = []
                formatted_result['actions'][player].append(action['action'])
            
            result.add_hand_result(formatted_result)
            
            if verbose and (hand_num + 1) % 10 == 0:
                print(f"Hand {hand_num + 1}/{num_hands} completed")
        
        return result
    
    def evaluate_agent(
        self,
        agent: BaseAgent,
        num_hands: int = 500,
        num_opponents: int = 5,
        opponent_type: str = 'random'
    ) -> Dict[str, float]:
        """
        Evaluate a single agent against opponents
        
        Args:
            agent: Agent to evaluate
            num_hands: Number of hands to play
            num_opponents: Number of opponent players
            opponent_type: Type of opponents ('random', 'call', 'fold')
            
        Returns:
            Dictionary with evaluation metrics
        """
        from src.agents.rule_agent import RuleBasedAgent
        
        # Register the agent
        self.agents = {0: agent}
        self.agent_names = {0: "Evaluated_Agent"}
        
        # Create opponents
        for i in range(1, num_opponents + 1):
            if opponent_type == 'random':
                from src.agents.rule_agent import RandomAgent
                opp = RandomAgent()
            elif opponent_type == 'call':
                opp = RuleBasedAgent(strategy='always_call')
            else:  # fold
                opp = RuleBasedAgent(strategy='always_fold')
            
            self.agents[i] = opp
            self.agent_names[i] = f"Opponent_{i}"
        
        # Run tournament
        self.num_players = num_opponents + 1
        result = self.run_tournament(num_hands=num_hands, verbose=False)
        
        return {
            'win_rate': result.get_win_rate('Evaluated_Agent'),
            'avg_profit': result.get_average_profit('Evaluated_Agent'),
            'total_profit': result.total_profit.get('Evaluated_Agent', 0),
            'num_hands': num_hands
        }
    
    def head_to_head(
        self,
        agent1: BaseAgent,
        agent2: BaseAgent,
        num_hands: int = 1000,
        name1: str = "Agent_A",
        name2: str = "Agent_B"
    ) -> Dict[str, float]:
        """
        Run head-to-head match between two agents
        
        Args:
            agent1: First agent
            agent2: Second agent
            num_hands: Number of hands to play
            name1: Name for first agent
            name2: Name for second agent
            
        Returns:
            Dictionary with head-to-head results
        """
        # Heads-up: 2 players
        self.num_players = 2
        self.agents = {0: agent1, 1: agent2}
        self.agent_names = {0: name1, 1: name2}
        
        result = self.run_tournament(num_hands=num_hands, verbose=False)
        
        return {
            f'{name1}_win_rate': result.get_win_rate(name1),
            f'{name2}_win_rate': result.get_win_rate(name2),
            f'{name1}_avg_profit': result.get_average_profit(name1),
            f'{name2}_avg_profit': result.get_average_profit(name2),
            'total_hands': num_hands,
            'tie_rate': 1 - result.get_win_rate(name1) - result.get_win_rate(name2)
        }
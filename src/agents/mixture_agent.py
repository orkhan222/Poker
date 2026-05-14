"""
Mixture of Experts and Ensemble agents
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from .base_agent import BaseAgent, AgentAction
from .policy_agent import PolicyAgent
from .rule_agent import RuleBasedAgent


class MixtureAgent(BaseAgent):
    """
    Mixture of Experts agent that combines multiple agents.
    Uses a gating network to decide which agent to trust.
    """
    
    def __init__(
        self,
        name: str = "MixtureAgent",
        agents: Optional[List[BaseAgent]] = None,
        agent_weights: Optional[List[float]] = None,
        use_confidence_weighting: bool = True
    ):
        """
        Initialize mixture agent.
        
        Args:
            name: Agent name
            agents: List of agent instances
            agent_weights: Fixed weights for each agent (if None, equal weights)
            use_confidence_weighting: Weight by agent confidence
        """
        super().__init__(name)
        
        self.agents = agents or []
        self.agent_weights = agent_weights or [1.0] * len(self.agents)
        self.use_confidence_weighting = use_confidence_weighting
        
        # Normalize weights
        total = sum(self.agent_weights)
        if total > 0:
            self.agent_weights = [w / total for w in self.agent_weights]
        
        # Agent performance tracking
        self.agent_performance = defaultdict(float)
        self.action_count = defaultdict(int)
    
    def add_agent(self, agent: BaseAgent, weight: float = 1.0):
        """Add an agent to the mixture"""
        self.agents.append(agent)
        self.agent_weights.append(weight)
        
        # Re-normalize
        total = sum(self.agent_weights)
        if total > 0:
            self.agent_weights = [w / total for w in self.agent_weights]
    
    def _get_weighted_action(self, state: Dict) -> Tuple[AgentAction, Dict]:
        """Get weighted action from all agents"""
        actions = []
        total_weight = 0.0
        
        for i, agent in enumerate(self.agents):
            try:
                action = agent.act(state)
                weight = self.agent_weights[i]
                
                if self.use_confidence_weighting:
                    weight *= action.confidence
                
                actions.append((action, weight))
                total_weight += weight
                
            except Exception as e:
                print(f"Agent {i} failed: {e}")
                continue
        
        if not actions or total_weight == 0:
            # Fallback to conservative rule
            rule_agent = RuleBasedAgent(strategy='conservative')
            return rule_agent.act(state), {}
        
        # Weighted voting
        action_votes = defaultdict(lambda: {'weight': 0.0, 'bet_sizes': [], 'confidences': []})
        
        for action, weight in actions:
            action_votes[action.action]['weight'] += weight
            action_votes[action.action]['bet_sizes'].append(action.bet_size)
            action_votes[action.action]['confidences'].append(action.confidence * weight)
        
        # Select action with highest weight
        best_action = max(action_votes.keys(), key=lambda a: action_votes[a]['weight'])
        vote_data = action_votes[best_action]
        
        # Calculate average bet size (weighted)
        if vote_data['bet_sizes']:
            bet_size = np.average(vote_data['bet_sizes'], weights=vote_data['confidences'])
        else:
            bet_size = 0.0
        
        confidence = min(1.0, vote_data['weight'] / total_weight)
        
        return AgentAction(best_action, bet_size, confidence), action_votes
    
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """Get action from mixture of agents"""
        action, votes = self._get_weighted_action(state)
        
        # Record
        self.action_history.append({
            'action': action.action,
            'bet_size': action.bet_size,
            'confidence': action.confidence,
            'votes': votes
        })
        
        return action
    
    def update_performance(self, action_taken: AgentAction, reward: float):
        """Update agent performance based on outcomes"""
        # This would be used for online learning of weights
        pass
    
    def reset(self):
        """Reset all agents"""
        for agent in self.agents:
            agent.reset()
        self.action_history = []


class EnsembleAgent(BaseAgent):
    """
    Ensemble agent that aggregates actions from multiple agents.
    Supports majority voting and averaging.
    """
    
    def __init__(
        self,
        name: str = "EnsembleAgent",
        agents: Optional[List[BaseAgent]] = None,
        voting_method: str = "majority"  # 'majority', 'weighted', 'confidence'
    ):
        """
        Initialize ensemble agent.
        
        Args:
            name: Agent name
            agents: List of agent instances
            voting_method: 'majority', 'weighted', 'confidence'
        """
        super().__init__(name)
        
        self.agents = agents or []
        self.voting_method = voting_method
        
        # Action mapping
        self.action_values = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all_in': 5}
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the ensemble"""
        self.agents.append(agent)
    
    def _majority_vote(self, actions: List[AgentAction]) -> AgentAction:
        """Simple majority voting"""
        votes = defaultdict(int)
        bet_sizes = defaultdict(list)
        
        for action in actions:
            votes[action.action] += 1
            bet_sizes[action.action].append(action.bet_size)
        
        winner = max(votes.keys(), key=lambda a: votes[a])
        avg_bet = np.mean(bet_sizes[winner]) if bet_sizes[winner] else 0.0
        
        return AgentAction(winner, avg_bet, confidence=votes[winner] / len(actions))
    
    def _weighted_vote(self, actions: List[AgentAction], weights: List[float]) -> AgentAction:
        """Weighted voting based on agent weights"""
        weighted_votes = defaultdict(float)
        weighted_bets = defaultdict(float)
        
        for action, weight in zip(actions, weights):
            weighted_votes[action.action] += weight
            weighted_bets[action.action] += action.bet_size * weight
        
        total_weight = sum(weights)
        if total_weight == 0:
            return self._majority_vote(actions)
        
        winner = max(weighted_votes.keys(), key=lambda a: weighted_votes[a])
        bet_size = weighted_bets[winner] / weighted_votes[winner] if weighted_votes[winner] > 0 else 0.0
        confidence = weighted_votes[winner] / total_weight
        
        return AgentAction(winner, bet_size, confidence)
    
    def _confidence_vote(self, actions: List[AgentAction]) -> AgentAction:
        """Vote based on action confidence"""
        confidence_scores = defaultdict(float)
        bet_sizes = defaultdict(list)
        
        for action in actions:
            confidence_scores[action.action] += action.confidence
            bet_sizes[action.action].append(action.bet_size)
        
        winner = max(confidence_scores.keys(), key=lambda a: confidence_scores[a])
        avg_bet = np.mean(bet_sizes[winner]) if bet_sizes[winner] else 0.0
        confidence = min(1.0, confidence_scores[winner] / len(actions))
        
        return AgentAction(winner, avg_bet, confidence)
    
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """Get ensemble action"""
        if not self.agents:
            # Fallback
            rule_agent = RuleBasedAgent(strategy='conservative')
            return rule_agent.act(state)
        
        # Get actions from all agents
        agent_actions = []
        for agent in self.agents:
            try:
                action = agent.act(state)
                agent_actions.append(action)
            except Exception as e:
                print(f"Ensemble agent error: {e}")
                continue
        
        if not agent_actions:
            rule_agent = RuleBasedAgent(strategy='conservative')
            return rule_agent.act(state)
        
        # Apply voting method
        if self.voting_method == 'majority':
            return self._majority_vote(agent_actions)
        elif self.voting_method == 'confidence':
            return self._confidence_vote(agent_actions)
        else:
            # Weighted - use equal weights if not specified
            weights = [1.0] * len(agent_actions)
            return self._weighted_vote(agent_actions, weights)
    
    def reset(self):
        """Reset all agents"""
        for agent in self.agents:
            agent.reset()
        self.action_history = []


class AdaptiveMixtureAgent(MixtureAgent):
    """
    Adaptive mixture agent that learns optimal agent weights over time.
    """
    
    def __init__(
        self,
        name: str = "AdaptiveMixture",
        agents: Optional[List[BaseAgent]] = None,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1
    ):
        super().__init__(name, agents)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.agent_rewards = defaultdict(float)
        self.agent_counts = defaultdict(int)
    
    def update_weights(self, reward: float, action_taken: AgentAction):
        """Update agent weights based on reward"""
        for i, agent in enumerate(self.agents):
            # Update average reward for this agent
            self.agent_rewards[i] += reward
            self.agent_counts[i] += 1
            
            # Update weight based on performance
            if self.agent_counts[i] > 0:
                avg_reward = self.agent_rewards[i] / self.agent_counts[i]
                # Normalize reward to [0, 1] range
                normalized = max(0.0, min(1.0, (avg_reward + 50) / 100))
                self.agent_weights[i] = (1 - self.learning_rate) * self.agent_weights[i] + \
                                         self.learning_rate * normalized
        
        # Re-normalize
        total = sum(self.agent_weights)
        if total > 0:
            self.agent_weights = [w / total for w in self.agent_weights]
    
    def act(self, state: Dict[str, Any]) -> AgentAction:
        """Get action with exploration"""
        # Exploration: random agent
        if np.random.random() < self.exploration_rate and self.agents:
            idx = np.random.randint(len(self.agents))
            action = self.agents[idx].act(state)
            self.action_history.append({
                'action': action.action,
                'bet_size': action.bet_size,
                'mode': 'exploration',
                'agent_idx': idx
            })
            return action
        
        # Exploitation: weighted mixture
        return super().act(state)
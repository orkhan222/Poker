"""
Reinforcement Learning trainers (PPO, A2C) for poker
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from tqdm import tqdm
import wandb
from pathlib import Path

from src.environment.poker_env import PokerEnv, PokerEnvConfig
from src.models.policy_net import PolicyNetwork
from src.models.value_net import ValueNetwork, DualHeadNetwork
from src.models.utils import ModelCheckpoint


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    num_episodes: int = 5000
    max_steps_per_episode: int = 200
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    batch_size: int = 64
    num_opponents: int = 5
    starting_stack: float = 1000.0
    use_gae: bool = True
    normalize_advantages: bool = True
    max_grad_norm: float = 0.5
    use_wandb: bool = False
    wandb_project: str = 'poker-rl'
    checkpoint_dir: str = 'experiments/checkpoints'
    save_interval: int = 100
    log_interval: int = 10
    device: str = 'cuda'


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for poker.
    Supports both separate policy/value networks or dual-head network.
    """
    
    def __init__(
        self,
        env: PokerEnv,
        policy_net: nn.Module,
        value_net: Optional[nn.Module] = None,
        config: PPOConfig = None,
        device: str = 'cuda'
    ):
        self.env = env
        self.config = config or PPOConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.policy_net = policy_net.to(self.device)
        
        if value_net is not None:
            self.value_net = value_net.to(self.device)
            self.use_dual_head = False
        else:
            # Check if policy_net has value head
            if hasattr(policy_net, 'value_head') or isinstance(policy_net, DualHeadNetwork):
                self.value_net = policy_net
                self.use_dual_head = True
            else:
                self.value_net = ValueNetwork().to(self.device)
                self.use_dual_head = False
        
        # Optimizer
        if self.use_dual_head:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                lr=self.config.learning_rate
            )
        
        # Checkpoint manager
        self.checkpointer = ModelCheckpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            save_best_only=True,
            monitor='episode_reward',
            mode='max'
        )
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def collect_trajectory(self) -> Dict[str, List]:
        """
        Collect one trajectory (episode) from the environment
        """
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        
        state, _ = self.env.reset()
        episode_reward = 0
        
        for step in range(self.config.max_steps_per_episode):
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get action and value
            with torch.no_grad():
                if self.use_dual_head:
                    action_logits, value = self.policy_net(state_tensor)
                    value = value.squeeze().cpu().item()
                else:
                    action_logits = self.policy_net(state_tensor)
                    value = self.value_net(state_tensor).squeeze().cpu().item()
                
                probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Take step in environment
            action_int = action.cpu().item()
            next_state, reward, done, truncated, info = self.env.step(action_int)
            
            # Store
            states.append(state)
            actions.append(action_int)
            rewards.append(reward)
            dones.append(done or truncated)
            values.append(value)
            log_probs.append(log_prob.cpu().item())
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        # Compute final value
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.use_dual_head:
                _, next_value = self.policy_net(state_tensor)
                next_value = next_value.squeeze().cpu().item()
            else:
                next_value = self.value_net(state_tensor).squeeze().cpu().item()
        
        # Compute advantages
        if self.config.use_gae:
            advantages = self.compute_gae(rewards, values, dones, next_value)
        else:
            # Simple return-based advantage
            returns = []
            running_return = next_value
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.config.gamma * running_return * (1 - dones[t])
                returns.insert(0, running_return)
            returns = np.array(returns)
            advantages = returns - np.array(values)
        
        if self.config.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = np.array(values) + advantages
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'returns': returns,
            'advantages': advantages,
            'old_log_probs': np.array(log_probs),
            'reward': episode_reward,
            'length': len(states)
        }
    
    def update_policy(self, trajectory: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """
        Update policy using PPO objective
        """
        states = torch.tensor(trajectory['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(trajectory['actions'], dtype=torch.long).to(self.device)
        returns = torch.tensor(trajectory['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(trajectory['advantages'], dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(trajectory['old_log_probs'], dtype=torch.float32).to(self.device)
        
        # Create dataset
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get current predictions
                if self.use_dual_head:
                    action_logits, values = self.policy_net(batch_states)
                else:
                    action_logits = self.policy_net(batch_states)
                    values = self.value_net(batch_states).squeeze(-1)
                
                # Compute probabilities
                probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        n_updates = self.config.update_epochs * (dataset_size // self.config.batch_size + 1)
        
        return total_policy_loss / n_updates, total_value_loss / n_updates, total_entropy / n_updates
    
    def train(self) -> Dict[str, List]:
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"PPO TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Episodes: {self.config.num_episodes}")
        print(f"Opponents: {self.config.num_opponents}")
        print(f"{'='*60}\n")
        
        # Initialize wandb
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    'num_episodes': self.config.num_episodes,
                    'learning_rate': self.config.learning_rate,
                    'clip_epsilon': self.config.clip_epsilon,
                    'gamma': self.config.gamma
                }
            )
        
        pbar = tqdm(range(1, self.config.num_episodes + 1), desc='Training')
        
        for episode in pbar:
            # Collect trajectory
            trajectory = self.collect_trajectory()
            
            # Update policy
            policy_loss, value_loss, entropy = self.update_policy(trajectory)
            
            # Store metrics
            self.episode_rewards.append(trajectory['reward'])
            self.episode_lengths.append(trajectory['length'])
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            self.entropies.append(entropy)
            
            # Update progress bar
            avg_reward = np.mean(self.episode_rewards[-100:])
            pbar.set_postfix({
                'reward': f'{trajectory["reward"]:.2f}',
                'avg_reward': f'{avg_reward:.2f}',
                'len': trajectory['length']
            })
            
            # Logging
            if episode % self.config.log_interval == 0:
                print(f"\nEpisode {episode}: Reward={trajectory['reward']:.2f}, "
                      f"Avg Reward={avg_reward:.2f}, Length={trajectory['length']}")
                print(f"  Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}, Entropy={entropy:.4f}")
            
            # Save checkpoint
            if episode % self.config.save_interval == 0:
                self.checkpointer.save(
                    self.policy_net,
                    self.optimizer,
                    episode,
                    {'episode_reward': avg_reward},
                    f'ppo_episode_{episode}'
                )
            
            # Wandb logging
            if self.config.use_wandb:
                wandb.log({
                    'episode': episode,
                    'reward': trajectory['reward'],
                    'avg_reward_100': avg_reward,
                    'episode_length': trajectory['length'],
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'entropy': entropy
                })
        
        # Save final model
        self.checkpointer.save(
            self.policy_net,
            self.optimizer,
            self.config.num_episodes,
            {'episode_reward': np.mean(self.episode_rewards[-100:])},
            'ppo_final'
        )
        
        if self.config.use_wandb:
            wandb.finish()
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropies': self.entropies,
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }


class A2CTrainer(PPOTrainer):
    """
    Advantage Actor-Critic (A2C) trainer for poker.
    Simplified version without PPO clipping.
    """
    
    def update_policy(self, trajectory: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """Update policy using A2C objective"""
        states = torch.tensor(trajectory['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(trajectory['actions'], dtype=torch.long).to(self.device)
        advantages = torch.tensor(trajectory['advantages'], dtype=torch.float32).to(self.device)
        
        # Forward pass
        if self.use_dual_head:
            action_logits, values = self.policy_net(states)
        else:
            action_logits = self.policy_net(states)
            values = self.value_net(states).squeeze(-1)
        
        # Compute losses
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        policy_loss = -(log_probs * advantages).mean()
        value_loss = nn.MSELoss()(values, advantages + values.detach())
        entropy = dist.entropy().mean()
        
        total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()


def train_rl(
    env: PokerEnv,
    policy_net: Optional[nn.Module] = None,
    value_net: Optional[nn.Module] = None,
    config: Optional[PPOConfig] = None,
    algorithm: str = 'ppo'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to train RL agent
    
    Args:
        env: Poker environment
        policy_net: Policy network
        value_net: Value network
        config: Training configuration
        algorithm: 'ppo' or 'a2c'
        
    Returns:
        (trained_policy, results) tuple
    """
    if policy_net is None:
        policy_net = PolicyNetwork()
    
    if config is None:
        config = PPOConfig()
    
    if algorithm.lower() == 'ppo':
        trainer = PPOTrainer(env, policy_net, value_net, config)
    elif algorithm.lower() == 'a2c':
        trainer = A2CTrainer(env, policy_net, value_net, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    results = trainer.train()
    return policy_net, results
"""
Self-play training for poker agents
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from tqdm import tqdm
import copy
import wandb
from pathlib import Path

from src.environment.simulator import MultiAgentSimulator, TournamentResult
from src.agents.policy_agent import PolicyAgent
from src.agents.rule_agent import RuleBasedAgent, RandomAgent
from src.models.policy_net import PolicyNetwork
from src.training.rl_trainer import PPOTrainer, PPOConfig


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training"""
    num_iterations: int = 100
    hands_per_iteration: int = 500
    update_frequency: int = 5
    opponent_pool_size: int = 10
    exploration_epsilon: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 64
    use_rule_opponents: bool = True
    num_rule_opponents: int = 3
    save_interval: int = 10
    eval_interval: int = 5
    num_eval_hands: int = 500
    checkpoint_dir: str = 'experiments/checkpoints'
    use_wandb: bool = False
    wandb_project: str = 'poker-selfplay'
    device: str = 'cuda'


class SelfPlayTrainer:
    """
    Self-play trainer for poker agents.
    Trains agent by playing against itself and past versions.
    """
    
    def __init__(
        self,
        config: SelfPlayConfig = None,
        device: str = 'cuda'
    ):
        self.config = config or SelfPlayConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Main agent
        self.main_agent = PolicyAgent(device=self.device)
        
        # Opponent pool (past versions + rule agents)
        self.opponent_pool: List[PolicyAgent] = []
        self.opponent_weights: List[float] = []
        
        # Rule-based opponents
        self.rule_opponents = []
        if self.config.use_rule_opponents:
            self.rule_opponents = [
                RuleBasedAgent(name="Conservative", strategy="conservative"),
                RuleBasedAgent(name="Aggressive", strategy="aggressive"),
                RuleBasedAgent(name="Tight", strategy="tight"),
                RuleBasedAgent(name="Loose", strategy="loose"),
                RandomAgent(name="Random")
            ]
        
        # Simulator
        self.simulator = MultiAgentSimulator(
            num_players=6,
            starting_stack=1000,
            verbose=False
        )
        
        # Training metrics
        self.win_rates = []
        self.elos = [1500]  # ELO rating history
        self.iteration_results = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _add_to_opponent_pool(self, agent: PolicyAgent):
        """Add agent to opponent pool"""
        # Deep copy the agent
        copied_agent = PolicyAgent(device=self.device)
        copied_agent.model.load_state_dict(agent.model.state_dict())
        copied_agent.model.eval()
        
        self.opponent_pool.append(copied_agent)
        self.opponent_weights.append(1.0)
        
        # Trim pool if too large
        if len(self.opponent_pool) > self.config.opponent_pool_size:
            # Remove oldest opponent
            self.opponent_pool.pop(0)
            self.opponent_weights.pop(0)
    
    def _select_opponent(self) -> PolicyAgent:
        """Select an opponent from the pool"""
        if not self.opponent_pool and not self.rule_opponents:
            # No opponents yet, use self
            return self.main_agent
        
        # Mix of rule opponents and past versions
        candidates = []
        weights = []
        
        # Add rule opponents
        for opp in self.rule_opponents:
            candidates.append(opp)
            weights.append(0.3)
        
        # Add past versions
        for opp in self.opponent_pool:
            candidates.append(opp)
            weights.append(0.7 / max(1, len(self.opponent_pool)))
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Select
        idx = np.random.choice(len(candidates), p=weights)
        return candidates[idx]
    
    def _run_evaluation(self) -> Dict[str, float]:
        """Evaluate current agent against various opponents"""
        results = {}
        
        # Against rule opponents
        for opp in self.rule_opponents:
            self.simulator.agents = {0: self.main_agent, 1: opp}
            self.simulator.agent_names = {0: "Main", 1: opp.name}
            
            tournament = self.simulator.run_tournament(
                num_hands=self.config.num_eval_hands,
                verbose=False
            )
            
            win_rate = tournament.get_win_rate("Main")
            avg_profit = tournament.get_average_profit("Main")
            results[f"vs_{opp.name}"] = {'win_rate': win_rate, 'avg_profit': avg_profit}
        
        # Against random baseline
        random_agent = RandomAgent()
        self.simulator.agents = {0: self.main_agent, 1: random_agent}
        self.simulator.agent_names = {0: "Main", 1: "Random"}
        
        tournament = self.simulator.run_tournament(
            num_hands=self.config.num_eval_hands,
            verbose=False
        )
        results["vs_Random"] = {
            'win_rate': tournament.get_win_rate("Main"),
            'avg_profit': tournament.get_average_profit("Main")
        }
        
        # Overall win rate
        avg_win_rate = np.mean([r['win_rate'] for r in results.values()])
        results['avg_win_rate'] = avg_win_rate
        
        return results
    
    def _update_elo(self, result: float, opponent_elo: int = 1500):
        """Update ELO rating based on match result"""
        K = 32
        current_elo = self.elos[-1]
        
        expected = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
        new_elo = current_elo + K * (result - expected)
        
        self.elos.append(new_elo)
        return new_elo
    
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Train for one self-play iteration
        """
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
        print(f"{'='*50}")
        
        # Select opponent
        opponent = self._select_opponent()
        print(f"Selected opponent: {opponent.name if hasattr(opponent, 'name') else 'Past Version'}")
        
        # Play hands against opponent
        self.simulator.agents = {0: self.main_agent, 1: opponent}
        self.simulator.agent_names = {0: "Main", 1: "Opponent"}
        
        tournament = self.simulator.run_tournament(
            num_hands=self.config.hands_per_iteration,
            verbose=False
        )
        
        # Get results
        win_rate = tournament.get_win_rate("Main")
        avg_profit = tournament.get_average_profit("Main")
        
        print(f"Result: Win Rate = {win_rate:.3f}, Avg Profit = ${avg_profit:.2f}")
        
        # Update ELO (assuming opponent ELO of 1500 for rule agents)
        opponent_elo = 1500
        self._update_elo(win_rate, opponent_elo)
        
        # Train main agent on collected experiences
        # This is where you'd implement actual RL updates
        # For now, we'll just track metrics
        
        # Add current version to opponent pool periodically
        if (iteration + 1) % self.config.update_frequency == 0:
            self._add_to_opponent_pool(self.main_agent)
            print(f"  -> Added current agent to opponent pool")
        
        # Run evaluation
        if (iteration + 1) % self.config.eval_interval == 0:
            eval_results = self._run_evaluation()
            print(f"\nEvaluation Results:")
            for name, metrics in eval_results.items():
                if name != 'avg_win_rate':
                    print(f"  {name}: WR={metrics['win_rate']:.3f}, Profit=${metrics['avg_profit']:.2f}")
            print(f"  Avg Win Rate: {eval_results['avg_win_rate']:.3f}")
            print(f"  ELO: {self.elos[-1]:.0f}")
        else:
            eval_results = {'avg_win_rate': win_rate}
        
        # Save checkpoint
        if (iteration + 1) % self.config.save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f'self_play_iter_{iteration + 1}.pt'
            torch.save({
                'iteration': iteration,
                'model_state_dict': self.main_agent.model.state_dict(),
                'win_rate': win_rate,
                'elo': self.elos[-1]
            }, checkpoint_path)
            print(f"  -> Saved checkpoint to {checkpoint_path}")
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log({
                'iteration': iteration + 1,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'elo': self.elos[-1],
                'opponent_pool_size': len(self.opponent_pool)
            })
        
        return {
            'iteration': iteration,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'elo': self.elos[-1],
            'eval_results': eval_results
        }
    
    def train(self, load_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run full self-play training"""
        print(f"\n{'='*60}")
        print(f"SELF-PLAY TRAINING")
        print(f"{'='*60}")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Hands per iteration: {self.config.hands_per_iteration}")
        print(f"Update frequency: {self.config.update_frequency}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Load checkpoint if provided
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint, map_location=self.device)
            self.main_agent.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")
        
        # Initialize wandb
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config={
                    'num_iterations': self.config.num_iterations,
                    'hands_per_iteration': self.config.hands_per_iteration,
                    'update_frequency': self.config.update_frequency
                }
            )
        
        # Training loop
        for iteration in range(self.config.num_iterations):
            result = self.train_iteration(iteration)
            self.iteration_results.append(result)
            self.win_rates.append(result['win_rate'])
        
        # Save final model
        final_path = self.checkpoint_dir / 'self_play_final.pt'
        torch.save({
            'model_state_dict': self.main_agent.model.state_dict(),
            'win_rates': self.win_rates,
            'elos': self.elos,
            'config': self.config
        }, final_path)
        print(f"\n✅ Final model saved to {final_path}")
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        final_eval = self._run_evaluation()
        for name, metrics in final_eval.items():
            if name != 'avg_win_rate':
                print(f"{name}: Win Rate={metrics['win_rate']:.3f}, Avg Profit=${metrics['avg_profit']:.2f}")
        print(f"Average Win Rate: {final_eval['avg_win_rate']:.3f}")
        print(f"Final ELO: {self.elos[-1]:.0f}")
        
        if self.config.use_wandb:
            wandb.finish()
        
        return {
            'win_rates': self.win_rates,
            'elos': self.elos,
            'iteration_results': self.iteration_results,
            'final_eval': final_eval
        }


def run_self_play(
    config: Optional[SelfPlayConfig] = None,
    load_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run self-play training
    
    Args:
        config: Self-play configuration
        load_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Training results
    """
    if config is None:
        config = SelfPlayConfig()
    
    trainer = SelfPlayTrainer(config)
    results = trainer.train(load_checkpoint)
    
    return results
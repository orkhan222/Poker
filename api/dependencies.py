"""
Dependency injection for FastAPI
"""

from typing import Dict, Any, Optional
import os
import torch
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.rule_agent import RuleBasedAgent
from src.agents.llm_agent import LLMAgent, LLMAgentConfig
from src.utils.logger import get_logger

logger = get_logger()

# Global instances
_agent: Optional[BaseAgent] = None
_model = None
_config: Dict[str, Any] = {}


def load_config() -> Dict[str, Any]:
    """Load agent configuration from environment or config file"""
    config = {
        "agent_type": os.environ.get("AGENT_TYPE", "policy"),
        "model_path": os.environ.get("MODEL_PATH", "experiments/checkpoints/best_model.pt"),
        "device": os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
        "llm_api_key": os.environ.get("OPENAI_API_KEY", None),
        "llm_use_local": os.environ.get("LLM_USE_LOCAL", "false").lower() == "true",
        "strategy": os.environ.get("AGENT_STRATEGY", "conservative")
    }
    return config


def initialize_agent() -> BaseAgent:
    """
    Initialize the poker agent based on configuration
    
    Returns:
        Initialized agent
    """
    global _agent, _config
    
    _config = load_config()
    agent_type = _config["agent_type"]
    
    logger.info(f"Initializing agent of type: {agent_type}")
    
    if agent_type == "policy":
        # Policy network agent
        agent = PolicyAgent(
            name="PolicyAgent",
            model_path=_config["model_path"] if Path(_config["model_path"]).exists() else None,
            device=_config["device"],
            deterministic=False  # Use sampling for variety
        )
        logger.info(f"Policy agent initialized on {_config['device']}")
        
    elif agent_type == "llm":
        # LLM-based agent
        llm_config = LLMAgentConfig(
            model=_config["llm_model"],
            api_key=_config["llm_api_key"],
            use_local=_config["llm_use_local"],
            temperature=0.7,
            few_shot_examples=True
        )
        agent = LLMAgent(name="LLMAgent", config=llm_config)
        logger.info(f"LLM agent initialized with model: {_config['llm_model']}")
        
    elif agent_type == "rule":
        # Rule-based agent
        agent = RuleBasedAgent(
            name="RuleAgent",
            strategy=_config["strategy"],
            aggression=0.5
        )
        logger.info(f"Rule agent initialized with strategy: {_config['strategy']}")
        
    else:
        # Default to rule-based
        logger.warning(f"Unknown agent type: {agent_type}, using rule-based")
        agent = RuleBasedAgent(name="DefaultAgent", strategy="conservative")
    
    _agent = agent
    return agent


def get_agent() -> BaseAgent:
    """
    Dependency to get the agent instance
    
    Returns:
        Agent instance
    """
    global _agent
    if _agent is None:
        _agent = initialize_agent()
    return _agent


def get_model():
    """
    Get the underlying model (if available)
    
    Returns:
        Model instance or None
    """
    global _model
    if _model is None and _agent is not None:
        if hasattr(_agent, 'model'):
            _model = _agent.model
    return _model


def get_agent_stats() -> Dict[str, Any]:
    """
    Get agent statistics
    
    Returns:
        Dictionary of statistics
    """
    global _agent
    if _agent is None:
        return {"error": "Agent not initialized"}
    
    return _agent.get_stats()


def shutdown_agent():
    """
    Cleanup function for agent shutdown
    """
    global _agent, _model
    logger.info("Shutting down agent...")
    _agent = None
    _model = None
    logger.info("Agent shutdown complete")


def update_agent_config(config_update: Dict[str, Any]):
    """
    Update agent configuration dynamically
    
    Args:
        config_update: Dictionary of configuration updates
    """
    global _config
    _config.update(config_update)
    logger.info(f"Agent config updated: {config_update}")
    
    # Reinitialize agent if needed
    if 'agent_type' in config_update:
        initialize_agent()


class RateLimiter:
    """
    Simple rate limiter for API endpoints
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def check(self):
        """Check if request is allowed"""
        import time
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests = [t for t in self.requests if t > minute_ago]
        
        if len(self.requests) >= self.requests_per_minute:
            return False
        
        self.requests.append(now)
        return True


# Create rate limiter instances
predict_rate_limiter = RateLimiter(requests_per_minute=300)
evaluate_rate_limiter = RateLimiter(requests_per_minute=10)
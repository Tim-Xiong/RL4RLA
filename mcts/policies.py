"""Policy strategies for MCTS action selection."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from domain.action import Action, ActionType


class IPolicyStrategy(ABC):
    """Interface for MCTS policy strategies."""
    
    @abstractmethod
    def get_action_priors(self, legal_actions: List[Action]) -> List[Tuple[Action, float]]:
        """
        Get prior probabilities for legal actions.
        
        Args:
            legal_actions: List of legal Action objects
            
        Returns:
            List of (action, probability) tuples
        """
        pass


class UniformPolicy(IPolicyStrategy):
    """Uniform random policy - gives equal probability to all actions."""
    
    def get_action_priors(self, legal_actions: List[Action]) -> List[Tuple[Action, float]]:
        """Return uniform probabilities for all legal actions."""
        if not legal_actions:
            return []
        
        uniform_prob = 1.0 / len(legal_actions)
        return [(action, uniform_prob) for action in legal_actions]


class BiasedPolicy(IPolicyStrategy):
    """Policy with bias towards certain operation types."""
    
    def __init__(self):
        # Weights for different operation types by ID
        self.operation_biases = {
            0: 1.5,   # Vector-vector add (common)
            1: 1.5,   # Vector-vector sub (common)
            2: 2.0,   # Vector dot product (useful)
            3: 2.5,   # Matrix-vector multiply (very important)
            4: 1.8,   # Vector-matrix multiply
            5: 2.0,   # Scalar-vector multiply (common)
            6: 1.3,   # Scalar division
            7: 1.0,   # Sketch (expensive)
            8: 1.2,   # Matrix-matrix multiply
            9: 0.8,   # HHQR (very expensive)
            10: 1.5,  # Triangular solve
            11: 0.7,  # Matrix inverse (expensive)
            12: 1.0,  # LSQR
            13: 1.2,  # Matrix transpose multiply variants
            14: 1.2,
            15: 0.9,  # Leverage score (expensive)
            16: 0.8,  # Subsampling
        }
    
    def get_action_priors(self, legal_actions: List[Action]) -> List[Tuple[Action, float]]:
        """Return biased probabilities based on operation types."""
        if not legal_actions:
            return []
        
        # Calculate weights based on operation types
        weights = []
        for action in legal_actions:
            op_id = action.operation.operator.id
            weight = self.operation_biases.get(op_id, 1.0)
            weights.append(weight)
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(legal_actions)] * len(legal_actions)
        else:
            probabilities = [w / total_weight for w in weights]
        
        return list(zip(legal_actions, probabilities))


class HeuristicPolicy(IPolicyStrategy):
    """Policy using domain-specific heuristics."""
    
    def __init__(self):
        # Prefer certain operation patterns
        self.preferred_patterns = {
            # Prefer operations that build up from previous results
            "build_on_results": 1.5,
            # Prefer operations in outer loop for preprocessing
            "outer_preprocessing": 1.3,
            # Prefer vector operations in inner loop for efficiency
            "inner_vectors": 1.2,
        }
    
    def get_action_priors(self, legal_actions: List[Action]) -> List[Tuple[Action, float]]:
        """Return probabilities based on algorithmic heuristics."""
        if not legal_actions:
            return []
        
        weights = []
        for action in legal_actions:
            weight = self._calculate_heuristic_weight(action)
            weights.append(weight)
        
        # Normalize
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(legal_actions)] * len(legal_actions)
        else:
            probabilities = [w / total_weight for w in weights]
        
        return list(zip(legal_actions, probabilities))
    
    def _calculate_heuristic_weight(self, action: Action) -> float:
        """Calculate heuristic weight for an action."""
        try:
            op_id = action.operation.operator.id
            operand1_id = action.operation.operand1.id
            operand2_id = action.operation.operand2.id
            target_id = action.operation.target.id
            loop_select = action.loop_select
            action_type = action.action_type
            
            weight = 1.0
            
            # Prefer matrix-vector operations
            if op_id in [3, 4]:  # MAT_VEC_MUL, VEC_MAT_MUL
                weight *= 1.5
            
            # Prefer operations in outer loop for expensive computations
            if loop_select == 0 and op_id in [7, 9, 11]:  # SKETCH, HHQR, MAT_INV
                weight *= 1.3
            
            # Prefer vector operations in inner loop
            if loop_select == 1 and op_id in [0, 1, 2, 5]:  # Vector operations
                weight *= 1.2
            
            # Prefer INSERT actions (the only action type now)
            if action_type == ActionType.INSERT:
                weight *= 1.1
            
            return weight
            
        except (ValueError, IndexError):
            return 1.0


class AdaptivePolicy(IPolicyStrategy):
    """Policy that adapts based on search history."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.action_rewards = {}  # Track average rewards for actions
        self.action_counts = {}   # Track how often actions are tried
    
    def get_action_priors(self, legal_actions: List[Action]) -> List[Tuple[Action, float]]:
        """Return probabilities based on historical performance."""
        if not legal_actions:
            return []
        
        weights = []
        for action in legal_actions:
            # Use historical average reward, with optimistic initialization
            avg_reward = self.action_rewards.get(action, 0.5)  # Optimistic default
            count = self.action_counts.get(action, 1)
            
            # Weight based on average reward and exploration bonus
            exploration_bonus = 1.0 / np.sqrt(count)
            weight = avg_reward + exploration_bonus
            weights.append(max(weight, 0.1))  # Minimum weight
        
        # Normalize
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        return list(zip(legal_actions, probabilities))
    
    def update_action_reward(self, action: Action, reward: float):
        """Update the reward history for an action."""
        if action not in self.action_rewards:
            self.action_rewards[action] = reward
            self.action_counts[action] = 1
        else:
            # Update running average
            count = self.action_counts[action]
            old_avg = self.action_rewards[action]
            new_avg = old_avg + self.learning_rate * (reward - old_avg)
            
            self.action_rewards[action] = new_avg
            self.action_counts[action] = count + 1


class EnsemblePolicy(IPolicyStrategy):
    """Ensemble of multiple policies."""
    
    def __init__(self, policies: List[Tuple[IPolicyStrategy, float]]):
        """
        Initialize ensemble.
        
        Args:
            policies: List of (policy, weight) tuples
        """
        self.policies = policies
        # Normalize weights
        total_weight = sum(weight for _, weight in policies)
        self.policies = [(policy, weight / total_weight) for policy, weight in policies]
    
    def get_action_priors(self, legal_actions: List[Action]) -> List[Tuple[Action, float]]:
        """Combine priors from all policies."""
        if not legal_actions:
            return []
        
        # Initialize combined probabilities
        combined_probs = {action: 0.0 for action in legal_actions}
        
        # Weight and combine policy priors
        for policy, policy_weight in self.policies:
            policy_priors = policy.get_action_priors(legal_actions)
            policy_dict = dict(policy_priors)
            
            for action in legal_actions:
                prob = policy_dict.get(action, 0.0)
                combined_probs[action] += policy_weight * prob
        
        # Normalize (should already be normalized, but safety check)
        total_prob = sum(combined_probs.values())
        if total_prob > 0:
            combined_probs = {action: prob / total_prob 
                            for action, prob in combined_probs.items()}
        
        return list(combined_probs.items())
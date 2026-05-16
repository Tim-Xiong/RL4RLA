"""Performance metrics and reward calculation."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ExecutionResult:
    """Results from algorithm execution."""

    success: bool
    final_solution: Optional[np.ndarray] = None
    loss: float = float('inf') # relative residual norm
    convergence_history: Optional[list] = None
    computational_cost: float = 0.0
    condition_number: Optional[float] = None
    max_residual_ratio: float = 1.0

    @property
    def converged(self) -> bool:
        """Check if algorithm converged."""
        return self.success and self.loss < 1e-14


@dataclass
class AlgorithmMetrics:
    """Comprehensive metrics for algorithm performance."""

    # Core metrics
    reward: float = 0.0
    loss: float = float('inf')
    computational_complexity: float = 0.0
    relative_complexity: float = 0.0

    # Detailed breakdown
    base_reward: float = 0.0
    decay_reward: float = 0.0
    complexity_reward: float = 0.0
    condition_reward: float = 0.0

    # Additional info
    execution_success: bool = False

    def calculate_total_reward(self,
                             base_weight: float,
                             decay_weight: float,
                             complexity_weight: float,
                             condition_weight: float) -> float:
        """Calculate weighted total reward."""
        weights = [base_weight, decay_weight, complexity_weight, condition_weight]
        rewards = [self.base_reward, self.decay_reward, self.complexity_reward, self.condition_reward]

        # normalize weights to sum to 1
        normalized_weights = [w / sum(weights) for w in weights]
        self.reward = sum(r * w for r, w in zip(rewards, normalized_weights))
        return self.reward

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "reward": self.reward,
            "loss": self.loss,
            "computational_complexity": self.computational_complexity,
            "relative_complexity": self.relative_complexity,
            "base_reward": self.base_reward,
            "decay_reward": self.decay_reward,
            "complexity_reward": self.complexity_reward, 
            "condition_reward": self.condition_reward,
            "execution_success": self.execution_success
        }

class RewardCalculator:
    """Calculates rewards for algorithm performance."""

    def __init__(self,
                 base_weight: float,
                 decay_weight: float,
                 complexity_weight: float,
                 condition_weight: float,
                 reward_type: str = "log",
                 iterative_sketch: bool = False,
                 condition_reward_only: bool = False):
        self.base_weight = base_weight
        self.decay_weight = decay_weight
        self.complexity_weight = complexity_weight
        self.condition_weight = condition_weight
        self.reward_type = reward_type
        self.iterative_sketch = iterative_sketch
        self.condition_reward_only = condition_reward_only

    def calculate_metrics(self,
                         execution_result: ExecutionResult,
                         system_rows: int,
                         system_cols: int,
                         max_iterations: int) -> AlgorithmMetrics:
        """Calculate comprehensive metrics from execution result."""

        metrics = AlgorithmMetrics()

        if not execution_result.success:
            return metrics  # Return default metrics for failed execution

        metrics.execution_success = True
        metrics.loss = execution_result.loss
        metrics.computational_complexity = execution_result.computational_cost

        # Calculate base reward (loss-based)
        metrics.base_reward = self._calculate_base_reward(execution_result.loss)

        # Calculate decay reward (convergence-based)
        metrics.decay_reward = self._calculate_decay_reward(execution_result.max_residual_ratio)

        # Calculate complexity reward
        metrics.complexity_reward = self._calculate_complexity_reward(
            execution_result.computational_cost,
            execution_result.loss,
            system_rows,
            system_cols,
            max_iterations
        )
        metrics.relative_complexity = self._calculate_relative_complexity(
            execution_result.computational_cost,
            system_rows,
            system_cols,
            max_iterations
        )

        # Calculate condition reward
        if execution_result.condition_number is not None:
            metrics.condition_reward = self._calculate_condition_reward(execution_result.condition_number)

        # Calculate total weighted reward
        if self.condition_reward_only:
            metrics.reward = metrics.condition_reward
        else:
            metrics.condition_reward = 0.0
            metrics.calculate_total_reward(
                base_weight=self.base_weight,
                decay_weight=self.decay_weight,
                complexity_weight=self.complexity_weight,
                condition_weight=self.condition_weight
            )

        return metrics

    def _calculate_base_reward(self, loss: float) -> float:
        """Calculate base reward from loss value."""
        high_loss = 0.1
        low_loss = 1e-6
        loss_ratio = np.log(high_loss) - np.log(loss)
        base_loss_ratio = np.log(high_loss) - np.log(low_loss)
        return np.clip(loss_ratio / base_loss_ratio, 0, 1)

    def _calculate_decay_reward(self, max_residual_ratio: float) -> float:
        """Calculate reward based on convergence behavior."""
        # 0 < max_residual_ratio < 1 --> reasonable convergence behavior
        # max_residual_ratio >= 1 --> residual loss increased at some point
        if np.isnan(max_residual_ratio) or max_residual_ratio == 0:
            return 0.0

        return np.clip(-np.log(max_residual_ratio), 0, 1)

    def _calculate_complexity_reward(self,
                                   computational_cost: float,
                                   loss: float,
                                   system_rows: int,
                                   system_cols: int,
                                   max_iterations: int) -> float:
        """Calculate reward for computational efficiency."""
        loss_threshold = 1e-1
        row_threshold = 10

        # if self.iterative_sketch or (loss < loss_threshold and system_rows > row_threshold):
        if self.iterative_sketch or loss < loss_threshold:
            relative_complexity = self._calculate_relative_complexity(
                computational_cost,
                system_rows,
                system_cols,
                max_iterations
            )
            return 1 / (1 + np.exp(-5 + relative_complexity) + 1e-4)

        return 0.0

    def _calculate_condition_reward(self, condition_number: float) -> float:
        """Calculate reward based on condition number improvement."""
        return min(3, 4 / (condition_number / 10000.0)) / 3 # normalize to 0-1

    def _calculate_relative_complexity(self,
                                    computational_cost: float,
                                    system_rows: int,
                                    system_cols: int,
                                    max_iterations: int) -> float:
        """Calculate relative complexity."""
        complexity_base = system_rows * system_cols * max_iterations
        return computational_cost / (complexity_base + 1e-4)

# class RewardCalculator:
#     """Calculates rewards for algorithm performance."""

#     def __init__(self,
#                  base_weight: float,
#                  decay_weight: float,
#                  complexity_weight: float,
#                  condition_weight: float,
#                  reward_type: str = "log",
#                  iterative_sketch: bool = False,
#                  condition_reward_only: bool = False):
#         self.base_weight = base_weight
#         self.decay_weight = decay_weight
#         self.complexity_weight = complexity_weight
#         self.condition_weight = condition_weight
#         self.reward_type = reward_type
#         self.iterative_sketch = iterative_sketch
#         self.condition_reward_only = condition_reward_only

#     def calculate_metrics(self,
#                          execution_result: ExecutionResult,
#                          system_rows: int,
#                          system_cols: int,
#                          max_iterations: int) -> AlgorithmMetrics:
#         """Calculate comprehensive metrics from execution result."""

#         metrics = AlgorithmMetrics()

#         if not execution_result.success:
#             return metrics  # Return default metrics for failed execution

#         metrics.execution_success = True
#         metrics.loss = execution_result.loss
#         metrics.computational_complexity = execution_result.computational_cost

#         # Calculate base reward (loss-based)
#         metrics.base_reward = self._calculate_base_reward(execution_result.loss)

#         # Calculate decay reward (convergence-based)
#         metrics.decay_reward = self._calculate_decay_reward(execution_result.max_residual_ratio)

#         # Calculate complexity reward
#         metrics.complexity_reward = self._calculate_complexity_reward(
#             execution_result.computational_cost,
#             execution_result.loss,
#             system_rows,
#             system_cols,
#             max_iterations
#         )

#         # Calculate condition reward
#         if execution_result.condition_number is not None:
#             metrics.condition_reward = self._calculate_condition_reward(execution_result.condition_number)

#         # Calculate total weighted reward
#         if self.condition_reward_only:
#             metrics.reward = metrics.condition_reward
#         else:
#             metrics.condition_reward = 0.0
#             metrics.calculate_total_reward(
#                 base_weight=self.base_weight,
#                 decay_weight=self.decay_weight,
#                 complexity_weight=self.complexity_weight,
#                 condition_weight=self.condition_weight
#             )

#         return metrics

#     def _calculate_base_reward(self, loss: float) -> float:
#         """Calculate base reward from loss value."""
#         if self.reward_type == "log":
#             if self.iterative_sketch:
#                 min_loss = 0.03 # this cap reward for low loss
#                 sensitivity = 0.1 # lower sensitivity reward extremely low loss more, reward high loss less
#             else:
#                 min_loss = 1e-13
#                 sensitivity = 1e-13

#             safe_loss = max(loss, min_loss)
#             return max(0, (-np.log(safe_loss)) / (-np.log(sensitivity)))

#         elif self.reward_type == "exp":
#             sensitivity = 10 # higher sensitivity reward low loss more, reward high loss much less
#             # 0 < base_reward < 1, lower the loss, higher the reward
#             return np.exp(-sensitivity * loss)
#         else:
#             raise ValueError(f"Unknown reward type: {self.reward_type}")

#     def _calculate_decay_reward(self, max_residual_ratio: float) -> float:
#         """Calculate reward based on convergence behavior."""
#         # 0 < max_residual_ratio < 1 --> reasonable convergence behavior
#         # max_residual_ratio >= 1 --> residual loss increased at some point
#         if np.isnan(max_residual_ratio) or max_residual_ratio == 0:
#             return 0.0
#         return max(0, -np.log(max_residual_ratio))

#     def _calculate_complexity_reward(self,
#                                    computational_cost: float,
#                                    loss: float,
#                                    system_rows: int,
#                                    system_cols: int,
#                                    max_iterations: int) -> float:
#         """Calculate reward for computational efficiency."""
#         loss_threshold = 1e-1
#         row_threshold = 10

#         if self.iterative_sketch or (loss < loss_threshold and system_rows > row_threshold):
#             complexity_base = system_rows * system_cols * max_iterations
#             complexity_score = computational_cost / (complexity_base + 1e-4)
#             return 2 / (1 + np.exp(-5 + complexity_score) + 1e-4)

#         return 0.0

#     def _calculate_condition_reward(self, condition_number: float) -> float:
#         """Calculate reward based on condition number improvement."""
#         return min(3, 4 / (condition_number / 10000.0))

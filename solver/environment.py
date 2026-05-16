"""Solver environment interface and implementation."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from functools import lru_cache

from solver.executor import AlgorithmExecutor
from solver.grammar import ActionGenerator
from solver.variables import AlgorithmState, AlgorithmStateManager
from domain.linear_system import LinearSystem
from domain.action import Action
from domain.metrics import AlgorithmMetrics, ExecutionResult, RewardCalculator
from infrastructure.linear_algebra.operations import LinearAlgebraOperations


class ISolverEnvironment(ABC):
    """Interface for solver environments."""

    @abstractmethod
    def get_legal_actions(self, algorithm_state: AlgorithmState) -> List[Action]:
        """Get list of legal actions for given algorithm state."""
        pass

    @abstractmethod
    def apply_action(self, algorithm_state: AlgorithmState, action: Action) -> AlgorithmState:
        """Apply an action to algorithm state and return new state."""
        pass

    @abstractmethod
    def is_terminal(self, algorithm_state: AlgorithmState) -> bool:
        """Check if algorithm state is terminal."""
        pass

    @abstractmethod
    def get_execution_result(self, algorithm_state: AlgorithmState) -> ExecutionResult:
        """Get execution result for given algorithm state."""
        pass

    @abstractmethod
    def get_reward_metrics(self, algorithm_state: AlgorithmState) -> AlgorithmMetrics:
        """Get reward metrics for given algorithm state."""
        pass

    @abstractmethod
    def get_reward(self, algorithm_state: AlgorithmState) -> float:
        """Get reward for given algorithm state."""
        pass

    @abstractmethod
    def get_condition_number(self) -> float:
        """Get condition number for the environment."""
        pass


class SolverEnvironment(ISolverEnvironment):
    """Main solver environment implementation with functools.lru_cache."""

    def __init__(self,
                 linear_system: LinearSystem,
                 setup_loop: bool,
                 forward_loop: bool,
                 update_loop: bool,
                 max_steps: int = 10,
                 max_iterations: int = 30,
                 learning_rate: float = 0.2,
                 lr_sweep: bool = False,
                 batch_size: int = 100,
                 random_seed: int = 42,
                 reward_calculator: Optional[RewardCalculator] = None,
                 sketch_matrix = None):

        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.lr_sweep = lr_sweep
        self.batch_size = batch_size
        self.setup_loop = setup_loop
        self.forward_loop = forward_loop
        self.update_loop = update_loop
        
        # Store linear system for use in services
        self.linear_system = linear_system
        
        # Initialize executor (stateless)
        self.executor = AlgorithmExecutor(
            linear_system=linear_system,
            linear_algebra_ops=LinearAlgebraOperations(sketch_matrix),
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            batch_size=batch_size,
            random_seed=random_seed
        )
        
        # Initialize reward calculator
        self.reward_calculator = reward_calculator or RewardCalculator(
            base_weight=1.0,
            decay_weight=10.0,
            complexity_weight=8.0,
            condition_weight=1.0
        )

        # Initialize action generator and state manager
        # Determine initial symmetric operands based on system type
        from domain.linear_system import SystemType
        initial_symmetric = set()
        if linear_system.system_type == SystemType.PSD:
            from domain.operation import OperandType
            initial_symmetric.add(OperandType.A)

        self.action_generator = ActionGenerator(initial_symmetric_operands=initial_symmetric)
        self.state_manager = AlgorithmStateManager()

    def get_legal_actions(self, algorithm_state: AlgorithmState) -> List[Action]:
        """Get list of legal actions for given algorithm state."""
        legal_actions = list(self._get_legal_actions_cached(algorithm_state))
        # # print all legal actions for debugging
        # print(f"Legal actions for {len(legal_actions)} actions:")
        # for action in legal_actions:
        #     print(action.get_readable_string())
        return legal_actions

    def apply_action(self, algorithm_state: AlgorithmState, action: Action) -> AlgorithmState:
        """Apply an action to algorithm state and return new state."""
        # Use the state manager to apply the action and handle variable updates
        return self.state_manager.apply_action_to_state(algorithm_state, action)

    def is_terminal(self, algorithm_state: AlgorithmState) -> bool:
        """Check if algorithm state is terminal."""
        if algorithm_state.step >= self.max_steps:
            return True

        legal_actions = self._get_legal_actions_cached(algorithm_state)
        if not legal_actions:
            return True

        execution_result = self.get_execution_result(algorithm_state)
        if execution_result.converged:
            return True

        return False

    def get_execution_result(self, algorithm_state: AlgorithmState) -> ExecutionResult:
        """Get execution result for given algorithm state."""
        return self.executor.execute_algorithm(algorithm_state.algorithm, self.lr_sweep)

    def get_reward_metrics(self, algorithm_state: AlgorithmState) -> AlgorithmMetrics:
        return self._calculate_metrics_cached(algorithm_state)

    def get_reward(self, algorithm_state: AlgorithmState) -> float:
        """Get reward for given algorithm state."""
        return self._calculate_metrics_cached(algorithm_state).reward

    def get_condition_number(self) -> float:
        """Get condition number for the environment."""
        return self.linear_system.condition_number

    @lru_cache(maxsize=500)
    def _get_legal_actions_cached(self, algorithm_state: AlgorithmState) -> Tuple[Action, ...]:
        """Cached legal actions generation - returns tuple for hashability."""
        legal_actions = self.action_generator.get_legal_actions(
            algorithm_state, self.setup_loop, self.forward_loop, self.update_loop
        )

        # Remove duplicates and return as tuple (hashable)
        unique_actions = tuple(set(legal_actions))
        return unique_actions

    @lru_cache(maxsize=200)
    def _calculate_metrics_cached(self, algorithm_state: AlgorithmState) -> AlgorithmMetrics:
        """Cached metrics calculation - depends on execution."""
        execution_result = self.get_execution_result(algorithm_state)
        return self.reward_calculator.calculate_metrics(
            execution_result=execution_result,
            system_rows=self.linear_system.num_rows,
            system_cols=self.linear_system.num_cols,
            max_iterations=self.max_iterations
        )

    def reset(self):
        """Reset environment and clear all caches."""
        self._get_legal_actions_cached.cache_clear()
        self._calculate_metrics_cached.cache_clear()

    def get_state(self) -> dict:
        """Get current environment state."""
        return {
            "max_steps": self.max_steps,
            "system_type": self.linear_system.system_type.value,
            "system_size": (self.linear_system.num_rows, self.linear_system.num_cols),
            "condition_number": self.linear_system.condition_number
        }

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring performance."""
        return {
            "legal_actions_cache": self._get_legal_actions_cached.cache_info()._asdict(),
            "metrics_cache": self._calculate_metrics_cached.cache_info()._asdict()
        }

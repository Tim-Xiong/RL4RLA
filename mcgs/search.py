"""MCGS search algorithm implementation."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from mcgs.graph import MCGSGraph
from mcts.policies import IPolicyStrategy, UniformPolicy
from solver.environment import ISolverEnvironment
from domain.action import Action
from domain.algorithm import Algorithm
from domain.operation import OperatorType, OperandType
from solver.variables import AlgorithmState


class MCGSSearcher:
    """Implements Monte Carlo Graph Search algorithm."""

    def __init__(self,
                 policy_strategy: Optional[IPolicyStrategy] = None,
                 c: float = 5.0,
                 score_fn: str = "puct", # puct, uct, ucd
                 n_playouts: int = 1000,
                 add_exploration_noise: bool = False,
                 dirichlet_alpha: float = 0.3,
                 noise_weight: float = 0.25,
                 graph_reuse: bool = True,
                 random_seed: Optional[int] = None,
                 initial_algorithm: Optional[AlgorithmState] = None,
                 target_algorithm: Optional[AlgorithmState] = None,
                 focus_operators: Optional[List[OperatorType]] = None,
                 early_stopping: bool = False,
                 stopping_factor: float = 1.0,
                 use_lucb: bool = False,
                 best_action_metric: str = "visits"):

        self.policy_strategy = policy_strategy or UniformPolicy()
        self.c = c
        self.score_fn = score_fn
        self.n_playouts = n_playouts
        self.add_exploration_noise = add_exploration_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_weight = noise_weight
        self.graph_reuse = graph_reuse
        self.focus_operators = focus_operators
        self.early_stopping = early_stopping
        self.stopping_factor = stopping_factor
        self.use_lucb = use_lucb
        self.best_action_metric = best_action_metric

        # Use initial algorithm state directly if provided
        initial_state = initial_algorithm

        self.graph = MCGSGraph(initial_state=initial_state)
        self.rng = np.random.default_rng(random_seed)

        self.target_algorithm = target_algorithm if target_algorithm is not None else AlgorithmState.from_dict({
            "setup_loop": [
                # [OperatorType.LEVERAGE_SCORE, OperandType.A, OperandType.NONE, OperandType.V3],
                # [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
                # [OperatorType.HHQR, OperandType.A, OperandType.NONE, OperandType.R1],
                # [OperatorType.SKETCH, OperandType.A, OperandType.NONE, OperandType.R1],
                # [OperatorType.HHQR, OperandType.R1, OperandType.NONE, OperandType.R1],
                # [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
                # [OperatorType.MAT_MAT_TRANS_MUL, OperandType.R1, OperandType.R1, OperandType.R1],
                # [OperatorType.LEVERAGE_SCORE, OperandType.A, OperandType.NONE, OperandType.V3],
            ],
            "forward_loop": [
                # [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
                # [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
                # [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
            ],
            "update_loop": [
                # [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1],
                # [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1]
            ]
        })

        # self.target_algorithm = AlgorithmState.from_dict({
        #     "setup_loop": [],
        #     "forward_loop": [
        #         [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
        #         [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
        #         [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        #     ],
        #     "update_loop": [
        #         [OperatorType.MAT_MAT_TRANS_MUL, OperandType.A, OperandType.A, OperandType.R1],
        #         [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
        #         [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1],
        #         [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1]
        #     ]
        # })

        # Statistics
        self.search_statistics = {
            "total_playouts": 0,
            "successful_playouts": 0,
            "terminal_states_reached": 0,
            "average_reward": 0.0,
            "find_target_algorithm": 0
        }
    
    def search(self, environment: ISolverEnvironment, temperature: float = 1.0,
               step_idx: int = 0, experiment_db=None, experiment_id: int = None,
               replay_rewards: Optional[List[float]] = None) -> Tuple[str, Dict[str, float]]:
        """
        Run MCGS search to find the best action.

        Args:
            environment: The solver environment
            temperature: Temperature for action selection
            step_idx: Current step index in multi-step search (for database logging)
            experiment_db: Optional ExperimentDatabase instance for logging
            experiment_id: Experiment ID for database logging
            replay_rewards: Optional list of pre-computed rewards for replay (deterministic reconstruction)

        Returns:
            Tuple of (best_action, action_probabilities)
        """

        # Add exploration noise to root if enabled
        if self.add_exploration_noise:
            self._add_exploration_noise_to_root(environment)

        # Run playouts
        total_reward = 0.0
        successful_playouts = 0

        for playout_idx in range(self.n_playouts):
            # Check early stopping criteria
            if self.early_stopping and self.graph.check_early_stop(self.stopping_factor, self.c, self.score_fn):
                print(f"Early stopping triggered at playout {playout_idx+1}/{self.n_playouts}")
                break
            
            # Check zero-reward termination
            if self.graph.check_zero_reward_termination():
                print(f"Zero-reward termination: All {len(self.graph.nodes)} nodes visited, all Q=0 at playout {playout_idx+1}/{self.n_playouts}")
                break

            # Run single playout
            reward = self._run_playout(environment, replay_rewards, playout_idx)

            if reward is not None:
                total_reward += reward
                successful_playouts += 1

                # Log to database if provided
                if experiment_db is not None and experiment_id is not None:
                    experiment_db.log_playout(experiment_id, step_idx, playout_idx, reward)

            # Print progress periodically
            if playout_idx % max(1, self.n_playouts // 100) == 0:
                progress = (playout_idx / self.n_playouts) * 100
                avg_reward = total_reward / max(1, successful_playouts)
                print(f"Search progress: {progress:.1f}% "
                      f"({playout_idx}/{self.n_playouts}), "
                      f"avg_reward: {avg_reward:.3f}")
                self.print_search_info()
        
        # Update statistics
        self.search_statistics.update({
            "total_playouts": self.n_playouts,
            "successful_playouts": successful_playouts,
            "average_reward": total_reward / max(1, successful_playouts)
        })
        
        # Get best action and probabilities
        best_action = self.graph.get_best_action(metric=self.best_action_metric)
        action_probs = self.graph.get_action_probabilities(temperature)

        if best_action is None:
            print("Something went wrong: best action is None")
        
        return best_action, action_probs
    
    def _run_playout(self, environment: ISolverEnvironment,
                    replay_rewards: Optional[List[float]] = None,
                    playout_idx: int = 0) -> Optional[float]:
        """
        Run a single MCGS playout.

        Args:
            environment: The solver environment
            replay_rewards: Optional list of pre-computed rewards for deterministic replay
            playout_idx: Current playout index (used to index into replay_rewards)
        """

        try:
            # Selection: traverse graph to leaf, path is list of (parent, action, child)
            if self.use_lucb:
                self.graph.refill_lucb_queue(self.c, self.score_fn)
            path = self.graph.select_leaf(self.c, self.score_fn)
            leaf_node = path[-1][-1] if path else self.graph.root

            # Get algorithm state directly from leaf node
            current_algorithm_state = leaf_node.algorithm_state

            # Check if terminal state
            is_terminal = environment.is_terminal(current_algorithm_state)

            if is_terminal:
                self.search_statistics["terminal_states_reached"] += 1
                if current_algorithm_state == self.target_algorithm:
                    self.search_statistics["find_target_algorithm"] += 1

                # Use pre-computed reward if replaying, otherwise compute it
                if replay_rewards is not None and playout_idx < len(replay_rewards):
                    reward = replay_rewards[playout_idx]
                else:
                    reward = environment.get_reward(current_algorithm_state)

                # Backpropagate terminal reward
                self.graph.backpropagate(path, reward)
                return reward

            # Expansion: get legal actions and expand if node is new
            legal_actions = environment.get_legal_actions(current_algorithm_state)

            if legal_actions:
                # Get action priors from policy
                action_priors_with_states = []
                policy_priors = self.policy_strategy.get_action_priors(legal_actions)
                action_priors_dict = dict(policy_priors)

                for action in legal_actions:
                    new_algorithm_state = environment.apply_action(current_algorithm_state, action)
                    prior_prob = action_priors_dict.get(action, 0.0)
                    action_priors_with_states.append((action, prior_prob, new_algorithm_state))

                self.graph.expand_node(leaf_node, action_priors_with_states)

                # Select a child for evaluation
                if leaf_node.children:
                    action, selected_child = leaf_node.select_child(self.c, self.score_fn)
                    current_algorithm_state = selected_child.algorithm_state
                    path.append((leaf_node, action, selected_child))

            # Simulation: run rollout from current algorithm state
            # Use pre-computed reward if replaying, otherwise run actual rollout
            if replay_rewards is not None and playout_idx < len(replay_rewards):
                rollout_reward = replay_rewards[playout_idx]
            else:
                rollout_reward = self._run_rollout(environment, current_algorithm_state)

            # Backpropagation: update all nodes in path
            final_reward = rollout_reward if rollout_reward is not None else 0.0
            self.graph.backpropagate(path, final_reward)
            
            return final_reward
            
        except (ValueError, KeyError, IndexError) as e:
            print(f"Playout failed: {e}")
            import traceback
            print(f"Full traceback:")
            traceback.print_exc()
            raise e
            return None
    
    def _run_rollout(self, environment: ISolverEnvironment, algorithm_state: Algorithm, max_depth: int = 1000) -> Optional[float]:
        """Run random rollout from given algorithm state to terminal."""
        
        current_state = algorithm_state
        
        for _ in range(max_depth):
            # Check if terminal
            is_terminal = environment.is_terminal(current_state)
            if is_terminal:
                if current_state == self.target_algorithm:
                    self.search_statistics["find_target_algorithm"] += 1
                reward = environment.get_reward(current_state)
                return reward
            
            legal_actions = environment.get_legal_actions(current_state)
            random_action = self.rng.choice(legal_actions)
            
            # Apply action to get new state
            current_state = environment.apply_action(current_state, random_action)
        
        # If reached max depth without terminal, get current reward
        reward = environment.get_reward(current_state)
        return reward

    def _add_exploration_noise_to_root(self, environment: ISolverEnvironment):
        """Add Dirichlet noise to root node for exploration."""

        # Ensure root is expanded
        if self.graph.root.is_leaf():
            legal_actions = environment.get_legal_actions(self.graph.root.algorithm_state)
            if legal_actions:
                action_priors_with_states = []
                policy_priors = self.policy_strategy.get_action_priors(legal_actions)
                action_priors_dict = dict(policy_priors)

                for action in legal_actions:
                    new_algorithm_state = environment.apply_action(self.graph.root.algorithm_state, action)
                    prior_prob = action_priors_dict.get(action, 1.0 / len(legal_actions))
                    action_priors_with_states.append((action, prior_prob, new_algorithm_state))

                self.graph.expand_node(self.graph.root, action_priors_with_states)

        # Add noise to children
        if self.graph.root.children:
            actions = list(self.graph.root.children.keys())
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))

            for action, noise_val in zip(actions, noise):
                child = self.graph.root.children[action]
                child.prior_prob = (
                    child.prior_prob * (1 - self.noise_weight) +
                    noise_val * self.noise_weight
                )
    
    def update_graph_root(self, last_action: Optional[Action]):
        """Update graph root after making a move."""
        if self.graph_reuse and last_action:
            # Check if the action exists as a representative action in any merged action
            action_exists = any(
                merged_action.representative_action == last_action
                for merged_action in self.graph.root.children.values()
            )
            if action_exists:
                self.graph.move_root(last_action)
            else:
                raise ValueError(f"Invalid last action {last_action}.")
        else:
            raise ValueError(f"Invalid last action {last_action}.")
    
    def get_search_statistics(self) -> Dict:
        """Get search statistics."""
        graph_stats = self.graph.get_graph_statistics()
        return {**self.search_statistics, **graph_stats}
    
    def print_search_info(self):
        """Print search information for debugging."""
        print("\n=== MCGS Search Info ===")
        print("Configuration:")
        print(f"  c: {self.c}")
        print(f"  score_fn: {self.score_fn}")
        print(f"  n_playouts: {self.n_playouts}")
        print(f"  exploration_noise: {self.add_exploration_noise}")

        stats = self.get_search_statistics()
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        self.graph.print_graph_statistics(self.c, self.score_fn, self.focus_operators, self.stopping_factor)

    def print_cache_info(self, environment):
        """Print cache performance statistics."""
        if hasattr(environment, 'get_cache_stats'):
            print("\n=== Cache Performance ===")
            cache_stats = environment.get_cache_stats()
            for cache_name, stats in cache_stats.items():
                hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) if (stats['hits'] + stats['misses']) > 0 else 0
                print(f"{cache_name}:")
                print(f"  Size: {stats['currsize']}/{stats['maxsize']}")
                print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
                print(f"  Hit Rate: {hit_rate:.1%}")


class MCGSPlayer:
    """High-level MCGS player interface."""

    def __init__(self,
                 searcher: Optional[MCGSSearcher] = None,
                 random_seed: Optional[int] = None,
                 initial_algorithm: Optional[AlgorithmState] = None,
                 **searcher_kwargs):
        """
        Initialize MCGS Player.
        
        Args:
            searcher: Pre-configured MCGSSearcher instance. If None, creates a new one.
            random_seed: Random seed for reproducibility
            initial_algorithm: Optional initial algorithm state to start search from
            **searcher_kwargs: Additional arguments to pass to MCGSSearcher if searcher is None
        """
        if searcher is not None:
            self.searcher = searcher
        else:
            # Pass initial_algorithm to searcher if provided
            if initial_algorithm is not None:
                searcher_kwargs['initial_algorithm'] = initial_algorithm
            if random_seed is not None:
                searcher_kwargs['random_seed'] = random_seed
            self.searcher = MCGSSearcher(**searcher_kwargs)
        
        self.rng = np.random.default_rng(random_seed)
    
    def run_search(self,
                   environment: ISolverEnvironment,
                   new_steps: int,
                   temperature: float = 0.0,
                   experiment_db=None,
                   experiment_id: int = None):
        """
        Run MCGS search.

        Args:
            environment: The solver environment
            new_steps: Number of steps to search
            temperature: Temperature for action selection
            experiment_db: Optional ExperimentDatabase instance for logging
            experiment_id: Experiment ID for database logging
        """
        for step in range(new_steps):
            action, _ = self.get_action(environment, temperature, return_probabilities=True,
                                       step_idx=step, experiment_db=experiment_db,
                                       experiment_id=experiment_id)
            if action is None:
                print(f"\nSearch stopped at step {step}: algorithm already converged or no legal actions")
                break
            print(f"\nReached step {step}: {action.get_readable_string()}")

        final_algorithm = self.get_current_algorithm()
        return final_algorithm

    def get_action(self,
                  environment: ISolverEnvironment,
                  temperature: float = 0.0,
                  return_probabilities: bool = False,
                  step_idx: int = 0,
                  experiment_db=None,
                  experiment_id: int = None):
        """
        Get action from MCGS search.

        Args:
            environment: Current environment state
            temperature: Temperature parameter (0 = greedy, higher = more random)
            return_probabilities: Whether to return action probabilities
            step_idx: Current step index (for database logging)
            experiment_db: Optional ExperimentDatabase instance for logging
            experiment_id: Experiment ID for database logging

        Returns:
            Action string, or (action_string, probabilities) if return_probabilities=True
        """

        # Run MCGS search
        most_visited_action, action_probs = self.searcher.search(environment, temperature,
                                                                 step_idx=step_idx,
                                                                 experiment_db=experiment_db,
                                                                 experiment_id=experiment_id)

        if temperature > 0 and action_probs:
            selected_action = self.rng.choice(list(action_probs.keys()), p=list(action_probs.values()))
        else:
            selected_action = most_visited_action

        # Update graph for next search (only if action is not None)
        if selected_action is not None:
            self.searcher.update_graph_root(selected_action)

        if return_probabilities:
            return selected_action, action_probs
        return selected_action
    
    def get_current_algorithm(self):
        """Get the current algorithm from the graph."""
        return self.searcher.graph.root.algorithm_state

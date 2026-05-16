"""MCTS tree implementation."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from domain.action import Action
from solver.variables import AlgorithmState


class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, algorithm_state: AlgorithmState, parent: Optional['MCTSNode'] = None, prior_prob: float = 1.0):
        self.parent = parent
        self.children: Dict[Action, 'MCTSNode'] = {}

        # Statistics
        self.visit_count: int = 0
        self.reward_sum: float = 0.0
        self.q_value: float = 0.0
        self.prior_prob: float = prior_prob

        # Associated action to this node (if not root)
        self.action: Optional[Action] = None
        
        # Store the complete algorithm state this node represents
        self.algorithm_state: AlgorithmState = algorithm_state

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    def is_fully_explored(self) -> bool:
        """Check if all children have been visited at least once."""
        if not self.children:
            return False
        return all(child.visit_count > 0 for child in self.children.values())

    def expand(self, action_priors: List[Tuple[Action, float, AlgorithmState]]):
        """Expand node by creating children for given actions with their resulting algorithm states."""
        for action, prior_prob, new_algorithm_state in action_priors:
            if action not in self.children:
                child = MCTSNode(algorithm_state=new_algorithm_state, parent=self, prior_prob=prior_prob)
                child.action = action
                self.children[action] = child

    def _get_exploration_bonus(self, c: float, score_fn: str) -> float:
        """Get exploration bonus for this node based on the specified scoring function.

        Args:
            c: Exploration constant
            score_fn: Scoring function ('puct' or 'uct')

        Returns:
            Exploration bonus value
        """
        if self.is_root():
            return 0.0

        if self.visit_count == 0:
            return float('inf')

        if score_fn == "puct":
            return c * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        elif score_fn == "uct":
            return c * math.sqrt(2 * math.log(self.parent.visit_count) / self.visit_count)
        else:
            raise ValueError(f"Unknown score function: {score_fn}")

    def _calculate_score(self, c: float, score_fn: str) -> float:
        """Calculate score for this node based on the specified scoring function.

        Args:
            c: Exploration constant
            score_fn: Scoring function ('puct' or 'uct')

        Returns:
            Score value (float, may be inf for unvisited nodes)
        """
        if self.is_root():
            return 0.0

        if self.visit_count == 0:
            return float('inf')

        exploitation = self.q_value
        exploration = self._get_exploration_bonus(c, score_fn)

        return exploitation + exploration

    def _get_sorted_children(self, c: float, score_fn: str) -> List[Tuple[Action, 'MCTSNode']]:
        """Get children sorted by score with deterministic tie-breaking.

        Sorting key: (score, -visit_count, action)
        - Higher scores first
        - When scores equal, prefer less visited nodes
        - When all equal, sort by action (deterministic ordering)
        """
        scored_children = [
            (action, child, child._calculate_score(c, score_fn))
            for action, child in self.children.items()
        ]
        sorted_children = sorted(
            scored_children,
            key=lambda x: (x[2], -x[1].visit_count, x[0]),
            reverse=True
        )
        return [(action, child) for action, child, _ in sorted_children]

    def select_child(self, c: float, score_fn: str = "puct") -> Tuple[Action, 'MCTSNode']:
        """Select child with highest score."""
        sorted_children = self._get_sorted_children(c, score_fn)
        if not sorted_children:
            raise ValueError("No children to select from")
        return sorted_children[0]

    def update(self, reward: float):
        """Update node statistics with new reward."""
        self.visit_count += 1
        # Running average update
        self.q_value += (reward - self.q_value) / self.visit_count
        self.reward_sum += reward

    def get_puct_score(self, c: float) -> float:
        """Calculate PUCT score for this node."""
        return self._calculate_score(c, "puct")

    def get_uct_score(self, c: float) -> float:
        """Calculate UCT score for this node."""
        return self._calculate_score(c, "uct")

    def get_leader_challenger(self, c: float, score_fn: str = "puct") -> Tuple[Tuple[Action, 'MCTSNode'], Tuple[Action, 'MCTSNode']]:
        """Get leader and challenger for LUCB.

        Returns:
            Tuple of ((leader_action, leader_node), (challenger_action, challenger_node))
        """
        if not self.children or len(self.children) < 2:
            return None, None

        # Leader: highest Q value (with tie-breaking)
        leader_action, leader_node = max(
            self.children.items(),
            key=lambda item: (item[1].q_value, -item[1].visit_count, item[0])
        )
        leader = (leader_action, leader_node)

        # Challenger: highest selection score among non-leaders (with tie-breaking)
        challenger_action, challenger_node = max(
            ((action, child) for action, child in self.children.items() if action != leader_action),
            key=lambda item: (
                item[1]._calculate_score(c, score_fn),
                -item[1].visit_count,
                item[0]
            )
        )
        challenger = (challenger_action, challenger_node)
        return leader, challenger

    def get_most_visited_action(self) -> Optional[Action]:
        """Get the action leading to the most visited child."""
        if not self.children:
            return None
        # Tie-breaking: visit count, Q-value, then deterministic action comparison
        return max(
            self.children.items(),
            key=lambda item: (item[1].visit_count, item[1].q_value, item[0])
        )[0]

    def get_most_valuable_action(self) -> Optional[Action]:
        """Get the action with highest Q value."""
        if not self.children:
            return None
        # Tie-breaking: Q-value, then deterministic action comparison
        return max(
            self.children.items(),
            key=lambda item: (item[1].q_value, item[0])
        )[0]

    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Action, float]:
        """Get action selection probabilities based on visit counts."""
        if not self.children:
            return {}
        
        if temperature == 0.0:
            # Greedy selection - all probability on most visited action
            most_visited_action = self.get_most_visited_action()
            return {action: 1.0 if action == most_visited_action else 0.0 
                   for action in self.children.keys()}

        # Softmax with temperature
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        log_visits = np.log(visit_counts + 1e-10) / temperature

        # Numerical stability
        log_visits = log_visits - np.max(log_visits)
        probs = np.exp(log_visits)
        probs = probs / np.sum(probs)

        return {action: prob for action, prob in zip(self.children.keys(), probs)}

    def __str__(self) -> str:
        """String representation."""
        return (f"MCTSNode(visits={self.visit_count}, "
                f"q_value={self.q_value:.3f}, "
                f"children={len(self.children)})")


class MCTSTree:
    """MCTS tree for algorithm search."""

    def __init__(self, initial_state: Optional[AlgorithmState] = None):
        self.root = MCTSNode(initial_state or AlgorithmState())
        self.total_simulations = 0
        self.lucb_queue = []  # [(action, child), ...] for LUCB
        
        # Zero-reward tracking
        self.unvisited_node_count = 1  # Root starts unvisited
        self.nonzero_q_nodes: set = set()
        self.all_nodes_visited_once = False

    def select_leaf(self, c: float, score_fn: str = "puct") -> List[MCTSNode]:
        """Select a leaf node following `score_fn` policy. Returns path from root to leaf."""
        path = []
        current = self.root

        # LUCB: Use queued action at root if available
        if len(self.lucb_queue) > 0 and not current.is_leaf():
            action, next_node = self.lucb_queue.pop(0)
            path.append(current)
            current = next_node

        while not current.is_leaf():
            path.append(current)
            action, current = current.select_child(c, score_fn)

        path.append(current)
        return path

    def expand_node(self, node: MCTSNode, action_priors: List[Tuple[Action, float, AlgorithmState]]):
        """Expand a node with new children."""
        children_before = len(node.children)
        node.expand(action_priors)
        children_after = len(node.children)
        self.unvisited_node_count += (children_after - children_before)

    def backpropagate(self, path: List[MCTSNode], reward: float):
        """Backpropagate reward through the path."""
        self.total_simulations += 1
        for node in reversed(path):
            self._update_node_with_tracking(node, reward)
    
    def _update_node_with_tracking(self, node: MCTSNode, reward: float):
        """Update node stats and track zero-reward conditions."""
        was_unvisited = (node.visit_count == 0)
        old_q = node.q_value
        
        node.update(reward)
        
        # Track first visit
        if was_unvisited and node.visit_count > 0:
            self.unvisited_node_count -= 1
            if self.unvisited_node_count == 0:
                self.all_nodes_visited_once = True
        
        # Track non-zero Q values
        EPSILON = 1e-10
        if abs(old_q) < EPSILON and abs(node.q_value) >= EPSILON:
            self.nonzero_q_nodes.add(node)
        elif abs(old_q) >= EPSILON and abs(node.q_value) < EPSILON:
            self.nonzero_q_nodes.discard(node)

    def get_best_action(self, metric: str = "visits") -> Optional[Action]:
        """Get the best action from root based on the specified metric."""
        if metric == "visits":
            return self.root.get_most_visited_action()
        elif metric == "q_value":
            return self.root.get_most_valuable_action()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Action, float]:
        """Get action probabilities from root."""
        return self.root.get_action_probabilities(temperature)

    def move_root(self, action: Action):
        """Move root to the child corresponding to the action."""
        if action not in self.root.children:
            raise ValueError(f"Action {action} not found in root children.")

        new_root = self.root.children[action]

        # Manual cleanup for immediate memory reclamation
        def cleanup_subtree(node):
            for child in node.children.values():
                cleanup_subtree(child)
            node.children.clear()
            node.parent = None

        # Clean up all other subtrees
        for child_action, child_node in self.root.children.items():
            if child_action != action:
                cleanup_subtree(child_node)

        # Set new root
        self.root = new_root
        self.root.parent = None

    def refill_lucb_queue(self, c: float, score_fn: str) -> None:
        """Refill queue with [leader, challenger] for LUCB."""
        if self.root.is_fully_explored() and len(self.lucb_queue) == 0:
            leader, challenger = self.root.get_leader_challenger(c, score_fn)
            if leader and challenger:
                self.lucb_queue = [leader, challenger]

    def check_early_stop(self, stopping_factor: float, c: float, score_fn: str) -> bool:
        """Check if early stopping criteria is met at root node."""
        if not self.root.is_fully_explored():
            return False

        leader, challenger = self.root.get_leader_challenger(c, score_fn)
        if not leader or not challenger:
            return False

        leader_node, challenger_node = leader[1], challenger[1]

        # LUCB stopping criterion: LCB(leader) > UCB(challenger)
        L = leader_node.q_value - stopping_factor * leader_node._get_exploration_bonus(c, score_fn)
        U = challenger_node.q_value + stopping_factor * challenger_node._get_exploration_bonus(c, score_fn)

        return L > U
    
    def check_zero_reward_termination(self) -> bool:
        """Check if all nodes visited but all rewards are zero.
        
        Returns True if:
        1. All nodes have been visited at least once
        2. All Q values are zero (no progress possible)
        """
        return self.all_nodes_visited_once and len(self.nonzero_q_nodes) == 0

    def get_tree_statistics(self) -> Dict:
        """Get statistics about the tree."""
        def count_nodes(node: MCTSNode, depth: int = 0) -> Tuple[int, int]:
            """Count total nodes and max depth."""
            count = 1
            max_depth = depth
            for child in node.children.values():
                child_count, child_max_depth = count_nodes(child, depth + 1)
                count += child_count
                max_depth = max(max_depth, child_max_depth)
            return count, max_depth
        
        total_nodes, max_depth = count_nodes(self.root)
        
        return {
            "total_nodes": total_nodes,
            "max_depth": max_depth,
            "root_visits": self.root.visit_count,
            "root_children": len(self.root.children),
            "total_simulations": self.total_simulations
        }
    
    def get_visit_stats(self) -> Dict:
        """Get visit statistics for all nodes."""
        stats = {}
        for action, node in self.root.children.items():
            stats[action] = node.visit_count
        return stats

    def print_tree_statistics(self, c: float, score_fn: str = "puct", focus_operators: Optional[List['OperatorType']] = None, stopping_factor: float = 1.0):
        """Print tree statistics for the specified scoring function."""
        from utils.statistics_display import (
            collect_action_statistics_mcts,
            display_statistics_table,
            display_summary_statistics,
            filter_by_operators
        )

        # Display parameters
        top = 5
        focus_top = 5
        if focus_operators is None:
            focus_operators = []

        if not self.root.children:
            print("No children to display statistics for.")
            return

        # Collect statistics
        all_stats = collect_action_statistics_mcts(self.root, c, score_fn, stopping_factor)

        # Calculate children counts for summary
        children_counts = [len(child.children) for child in self.root.children.values()]

        # Print header
        score_name = score_fn.upper()
        print("\n" + "="*152)
        print(f"TREE STATISTICS ({score_name})")
        print("="*152)

        # Table 1: Ranked by visits
        display_statistics_table(all_stats, "Actions by Visits", "visits", score_fn, top, show_node_visits=False)

        # Table 2: Ranked by Q value (NEW)
        display_statistics_table(all_stats, "Actions by Q Value", "q_value", score_fn, top, show_node_visits=False)

        # Focus operators table
        if focus_operators:
            focus_stats = filter_by_operators(all_stats, focus_operators)
            if focus_stats:
                display_statistics_table(focus_stats, "Focus Operators", "visits", score_fn, focus_top, show_node_visits=False)

        # Summary statistics
        display_summary_statistics(all_stats, children_counts, score_fn, show_node_visits=False)

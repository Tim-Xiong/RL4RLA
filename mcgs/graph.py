"""MCGS graph implementation."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from domain.action import Action
from domain.operation import OperatorType
from solver.variables import AlgorithmState


class MergedAction:
    """Represents a group of actions that lead to the same child node."""

    def __init__(self, representative_action: Action, prior_prob: float):
        self.representative_action = representative_action
        self.original_actions: List[Action] = [representative_action]
        self.total_visits: int = 0
        self.action_visits: Dict[Action, int] = {representative_action: 0}
        self.prior_prob: float = prior_prob

    def add_action(self, action: Action, prior_prob: float):
        """Add another action to this merged group."""
        if action not in self.original_actions:
            self.original_actions.append(action)
            self.action_visits[action] = 0
            # Average the prior probabilities
            total_actions = len(self.original_actions)
            self.prior_prob = (self.prior_prob * (total_actions - 1) + prior_prob) / total_actions

    def increment_visits(self, action: Action):
        """Increment visit count for a specific action and total."""
        if action in self.action_visits:
            self.action_visits[action] += 1
            self.total_visits += 1

    def get_merged_count(self) -> int:
        """Get the number of actions merged in this group."""
        return len(self.original_actions)

    def get_display_string(self) -> str:
        """Get display string showing representative action and merge count."""
        if len(self.original_actions) == 1:
            return self.representative_action.get_readable_string()
        else:
            return f"{self.representative_action.get_readable_string()} (+{len(self.original_actions)-1})"


class MCGSNode:
    """A node in the MCGS graph that can have multiple parents."""
    
    def __init__(self, algorithm_state: AlgorithmState, parent: Optional['MCGSNode'] = None,
                 prior_prob: float = 1.0):
        # Map each parent to a list of actions that reach this child (handles redundant edges)
        self.parents: List['MCGSNode'] = [] if parent is None else [parent]
        self.children: Dict['MCGSNode', MergedAction] = {}

        self.N: int = 0  # Total visits to this node
        self.U: float = 0.0  # Utility estimate for this state
        self.Q: float = 0.0  # Expected utility under current policy
        self.prior_prob: float = prior_prob
        
        # Store the complete algorithm state this node represents
        self.algorithm_state: AlgorithmState = algorithm_state
        
        # Graph properties
        self.is_expanded: bool = False
        self.is_transposition: bool = False

    def add_parent(self, parent: 'MCGSNode'):
        """Add an additional parent (creates transposition node)."""
        if parent not in self.parents:
            self.parents.append(parent)
        self.is_transposition = len(self.parents) > 1

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return not self.is_expanded
    
    def is_fully_explored(self) -> bool:
        """Check if all children have been visited at least once."""
        if not self.is_expanded or not self.children:
            return False
        return all(child.N > 0 for child in self.children.keys())

    def expand(self, action_priors: List[Tuple[Action, float, AlgorithmState]], graph: 'MCGSGraph'):
        """Expand node by creating children for given actions with their resulting algorithm states."""
        for action, prior_prob, new_algorithm_state in action_priors:
            # Check for transposition in graph
            if new_algorithm_state in graph.nodes:
                # Reuse existing node (transposition)
                child = graph.nodes[new_algorithm_state]
                child.add_parent(self)
            else:
                # Create new node
                child = MCGSNode(algorithm_state=new_algorithm_state,
                                 parent=self, prior_prob=prior_prob)
                graph.nodes[new_algorithm_state] = child
            
            # Check if we already have a merged action to this child
            if child in self.children:
                # Add action to existing merged action
                self.children[child].add_action(action, prior_prob)
            else:
                # Create new merged action
                merged_action = MergedAction(action, prior_prob)
                self.children[child] = merged_action
        
        self.is_expanded = True

    def select_child(self, c: float, score_fn: str = "puct") -> Tuple[Action, 'MCGSNode']:
        """Select child with highest score."""
        if score_fn == "puct":
            uct_scores = self.get_puct_scores(c)
            child_node = uct_scores[0][0]
            merged_action = self.children[child_node]
            representative_action = merged_action.representative_action
        elif score_fn == "uct":
            uct_scores = self.get_uct_scores(c)
            child_node = uct_scores[0][0]
            merged_action = self.children[child_node]
            representative_action = merged_action.representative_action
        elif score_fn == "ucd":
            ucd_scores = self.get_ucd_scores(c)
            child_node = ucd_scores[0][0]
            merged_action = self.children[child_node]
            representative_action = merged_action.representative_action
        else:
            raise ValueError(f"Unknown score function: {score_fn}")
        return representative_action, child_node

    def get_total_action_visits(self) -> int:
        """Total visits to all children."""
        return sum(merged_action.total_visits for merged_action in self.children.values())

    def update_stats(self, value: float):
        """Update statistics using MCGS recursive formula."""
        self.U += value
        if self.is_leaf():
            self.N += 1
            self.Q += (value - self.Q) / self.N
        else:
            total_action_visits = self.get_total_action_visits()
            self.N = 1 + total_action_visits
            if self.N == 1:
                self.Q = value
            else:
                child_contribution = sum(
                    merged_action.total_visits * child.Q
                    for child, merged_action in self.children.items()
                )
                self.Q = (value + child_contribution) / self.N

    def _get_exploration_bonus(self, child: 'MCGSNode', score_fn: str) -> float:
        """Get exploration bonus for a child node based on the specified scoring function."""
        merged_action = self.children[child]

        if score_fn == "puct":
            visit_count = merged_action.total_visits
            return merged_action.prior_prob * math.sqrt(self.N) / (1 + visit_count)
        elif score_fn == "uct":
            visit_count = merged_action.total_visits
            return math.sqrt(math.log(self.N) / visit_count)
        elif score_fn == "ucd":
            return math.sqrt(math.log(self.N) / child.N)
        else:
            raise ValueError(f"Unknown score function: {score_fn}")

    def _calculate_child_score(self, child: 'MCGSNode', c: float, score_fn: str) -> float:
        """Calculate score for a child node based on the specified scoring function.

        Args:
            child: Child node to calculate score for
            c: Exploration constant
            score_fn: Scoring function ('puct', 'uct', or 'ucd')

        Returns:
            Score value (float, may be inf for unvisited nodes)
        """
        action_visit = self.children[child].total_visits
        
        if score_fn in ["puct", "uct", "ucd"]:
            if child.N == 0 or action_visit == 0:
                return float('inf')
            exploitation = child.Q
            exploration = self._get_exploration_bonus(child, score_fn)
            return exploitation + c * exploration
        else:
            raise ValueError(f"Unknown score function: {score_fn}")

    def _get_sorted_children(self, c: float, score_fn: str) -> List[Tuple['MCGSNode', float]]:
        """Get children sorted by score with deterministic tie-breaking.

        Sorting key: (score, -visit_count, action)
        - Higher scores first
        - When scores equal, prefer less visited nodes
        - When all equal, sort by action (deterministic ordering)
        """
        scored_children = [
            (child, self._calculate_child_score(child, c, score_fn))
            for child in self.children.keys()
        ]
        return sorted(
            scored_children,
            key=lambda x: (x[1], -x[0].N, self.children[x[0]].representative_action),
            reverse=True
        )

    def get_leader_challenger(self, c: float, score_fn: str = "puct") -> Tuple[Tuple[Action, 'MCGSNode'], Tuple[Action, 'MCGSNode']]:
        """Get leader and challenger for LUCB."""
        if not self.children or len(self.children) < 2:
            return None, None

        # Leader: highest Q value (with tie-breaking as in original sort)
        leader_node = max(
            self.children.keys(),
            key=lambda child: (child.Q, -child.N, self.children[child].representative_action)
        )
        leader_action = self.children[leader_node].representative_action
        leader = (leader_action, leader_node)

        # Challenger: highest selection score among non-leaders (with tie-breaking)
        challenger_candidates = [child for child in self.children.keys() if child != leader_node]
        challenger_node = max(
            challenger_candidates,
            key=lambda child: (
                self._calculate_child_score(child, c, score_fn),
                -child.N,
                self.children[child].representative_action
            )
        )
        challenger_action = self.children[challenger_node].representative_action
        challenger = (challenger_action, challenger_node)
        return leader, challenger

    def get_puct_scores(self, c: float) -> List[Tuple['MCGSNode', float]]:
        """Calculate PUCT score for children from this node."""
        return self._get_sorted_children(c, "puct")

    def get_uct_scores(self, c: float) -> List[Tuple['MCGSNode', float]]:
        """Calculate UCT score for children from this node."""
        return self._get_sorted_children(c, "uct")

    def get_ucd_scores(self, c: float) -> List[Tuple['MCGSNode', float]]:
        """Calculate UCD (DAG-aware UCT) score for children from this node."""
        return self._get_sorted_children(c, "ucd")

    def get_most_visited_action(self) -> Optional[Action]:
        """Get the action with highest visit count."""
        if not self.children:
            return None
        # Tie-breaking: visit count, Q-value, then deterministic action comparison
        most_visited_child = max(
            self.children.items(),
            key=lambda item: (item[1].total_visits, item[0].Q, item[1].representative_action)
        )[0]
        return self.children[most_visited_child].representative_action
    
    def get_most_valuable_action(self) -> Optional[Action]:
        """Get the action with highest Q value."""
        if not self.children:
            return None
        # Tie-breaking: Q-value, then deterministic action comparison
        most_valuable_child = max(
            self.children.items(),
            key=lambda item: (item[0].Q, item[1].representative_action)
        )[0]
        return self.children[most_valuable_child].representative_action

    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[Action, float]:
        """Get action selection probabilities based on visit counts."""
        if not self.children:
            return {}
        
        if temperature == 0.0:
            # Greedy selection - all probability on most visited action
            most_visited_action = self.get_most_visited_action()
            return {merged_action.representative_action: 1.0 if merged_action.representative_action == most_visited_action else 0.0
                   for merged_action in self.children.values()}

        # Softmax with temperature
        visit_counts = np.array([merged_action.total_visits for merged_action in self.children.values()])
        log_visits = np.log(visit_counts + 1e-10) / temperature

        # Numerical stability
        log_visits = log_visits - np.max(log_visits)
        probs = np.exp(log_visits)
        probs = probs / np.sum(probs)

        return {merged_action.representative_action: prob for merged_action, prob in zip(self.children.values(), probs)}

    def __str__(self) -> str:
        """String representation."""
        return (f"MCGSNode(N={self.N}, "
                f"Q={self.Q:.3f}, "
                f"U={self.U:.3f}, "
                f"children={len(self.children)}, "
                f"parents={len(self.parents)})")


class MCGSGraph:
    """MCGS graph for algorithm search."""
    
    def __init__(self, initial_state: Optional[AlgorithmState] = None):
        self.root = MCGSNode(initial_state or AlgorithmState())
        self.nodes: Dict[AlgorithmState, MCGSNode] = {self.root.algorithm_state: self.root}
        self.total_simulations = 0
        self.lucb_queue = [] # [leader, challenger]
        
        # Zero-reward tracking
        self.unvisited_node_count = 1  # Root starts unvisited
        self.nonzero_q_nodes: set = set()
        self.all_nodes_visited_once = False

    def select_leaf(self, c: float, score_fn: str = "puct") -> List[Tuple['MCGSNode', Action, 'MCGSNode']]:
        """Select a leaf node following `score_fn` policy.
        Returns a path as a list of (parent, action, child) tuples.
        If the root is a leaf, returns an empty list.
        """
        path: List[Tuple[MCGSNode, Action, MCGSNode]] = []
        current = self.root

        if len(self.lucb_queue) > 0 and not current.is_leaf():
            root_action, next_node = self.lucb_queue.pop(0)
            path.append((current, root_action, next_node))
            current = next_node

        while not current.is_leaf():
            action, next_node = current.select_child(c, score_fn)
            path.append((current, action, next_node))
            current = next_node

        return path

    def expand_node(self, node: MCGSNode, action_priors: List[Tuple[Action, float, AlgorithmState]]):
        """Expand a node with new children."""
        nodes_before = len(self.nodes)
        node.expand(action_priors, self)
        nodes_after = len(self.nodes)
        self.unvisited_node_count += (nodes_after - nodes_before)

    def backpropagate(self, path: List[Tuple['MCGSNode', Action, 'MCGSNode']], reward: float):
        """Backpropagate reward through the path using MCGS update rule.
        Path is a list of (parent, action, child) tuples.
        """

        if len(path) == 0:
            return
        
        self.total_simulations += 1
        
        # Increment action visit count for all (parent, action, child) in the path
        for parent, action, child in path:
            if child in parent.children:
                parent.children[child].increment_visits(action)

        # Update stats for all nodes in path (leaf first, then backwards)
        leaf_node = path[-1][-1]
        self._update_node_with_tracking(leaf_node, reward)

        for parent, action, child in reversed(path):
            self._update_node_with_tracking(parent, reward)
            # For transposition nodes, also update other parent paths
            if child.is_transposition:
                for other_parent in child.parents:
                    if other_parent != parent:
                        self._update_node_with_tracking(other_parent, reward)
    
    def _update_node_with_tracking(self, node: MCGSNode, reward: float):
        """Update node stats and track zero-reward conditions."""
        was_unvisited = (node.N == 0)
        old_q = node.Q
        
        node.update_stats(reward)
        
        # Track first visit
        if was_unvisited and node.N > 0:
            self.unvisited_node_count -= 1
            if self.unvisited_node_count == 0:
                self.all_nodes_visited_once = True
        
        # Track non-zero Q values
        EPSILON = 1e-10
        if abs(old_q) < EPSILON and abs(node.Q) >= EPSILON:
            self.nonzero_q_nodes.add(node)
        elif abs(old_q) >= EPSILON and abs(node.Q) < EPSILON:
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
        # Find the child that has this action as representative
        target_child = None
        for child, merged_action in self.root.children.items():
            if merged_action.representative_action == action:
                target_child = child
                break
        
        if target_child is None:
            raise ValueError(f"Action {action} not found in root children.")
        
        # Set new root
        self.root = target_child
        # Remove this node from its parents to make it a root
        self.root.parents = []
        self.root.is_transposition = False

    def refill_lucb_queue(self, c: float, score_fn: str) -> None:
        """Refill queue with [leader, challenger] for LUCB."""
        if self.root.is_fully_explored() and len(self.lucb_queue) == 0:
            leader, challenger = self.root.get_leader_challenger(c, score_fn)
            self.lucb_queue = [leader, challenger]

    def check_early_stop(self, stopping_factor: float, c: float, score_fn: str) -> bool:
        """Check if early stopping criteria is met at root node."""
        if not self.root.is_fully_explored():
            return False
        
        leader, challenger = self.root.get_leader_challenger(c, score_fn)
        leader_node, challenger_node = leader[1], challenger[1]
        
        L = leader_node.Q - stopping_factor * c * self.root._get_exploration_bonus(leader_node, score_fn)
        U = challenger_node.Q + stopping_factor * c * self.root._get_exploration_bonus(challenger_node, score_fn)

        return L > U
    
    def check_zero_reward_termination(self) -> bool:
        """Check if all nodes visited but all rewards are zero.
        
        Returns True if:
        1. All nodes have been visited at least once
        2. All Q values are zero (no progress possible)
        """
        return self.all_nodes_visited_once and len(self.nonzero_q_nodes) == 0

    def get_graph_statistics(self) -> Dict:
        """Get statistics about the graph."""
        total_nodes = len(self.nodes)
        transposition_nodes = sum(1 for node in self.nodes.values() if node.is_transposition)

        return {
            "total_nodes": total_nodes,
            "transposition_nodes": transposition_nodes,
            "transposition_ratio": transposition_nodes / max(1, total_nodes),
            "root_visits": self.root.N,
            "root_children": len(self.root.children),
            "total_simulations": self.total_simulations
        }

    def print_graph_statistics(self, c: float, score_fn: str = "puct", focus_operators: Optional[List['OperatorType']] = None, stopping_factor: float = 1.0):
        """Print graph statistics for the specified scoring function."""
        from utils.statistics_display import (
            collect_action_statistics_mcgs,
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
        all_stats = collect_action_statistics_mcgs(self.root, c, score_fn, stopping_factor)

        # Calculate children counts for summary
        children_counts = [len(child.children) for child in self.root.children.keys()]

        # Print header
        score_name = score_fn.upper()
        print("\n" + "="*160)
        print(f"GRAPH STATISTICS ({score_name})")
        print("="*160)

        # Table 1: Ranked by visits
        display_statistics_table(all_stats, "Actions by Visits", "visits", score_fn, top, show_node_visits=True)

        # Table 2: Ranked by Q value (NEW)
        display_statistics_table(all_stats, "Actions by Q Value", "q_value", score_fn, top, show_node_visits=True)

        # Focus operators table
        if focus_operators:
            focus_stats = filter_by_operators(all_stats, focus_operators)
            if focus_stats:
                display_statistics_table(focus_stats, "Focus Operators", "visits", score_fn, focus_top, show_node_visits=True)

        # Summary statistics
        display_summary_statistics(all_stats, children_counts, score_fn, show_node_visits=True)

    def print_node_children_statistics(self, action_str: str, c: float, score_fn: str = "puct", stopping_factor: float = 1.0):
        """Print statistics for children of a specific node identified by action string."""
        from utils.statistics_display import (
            collect_action_statistics_mcgs,
            display_statistics_table,
            display_summary_statistics
        )

        top = 10

        # Find the node corresponding to the action string
        target_node = None
        for child, merged_action in self.root.children.items():
            if merged_action.get_display_string() == action_str:
                target_node = child
                break

        if target_node is None:
            print(f"Action '{action_str}' not found in root children.")
            return

        if not target_node.children:
            print(f"Node for action '{action_str}' has no children.")
            return

        # Collect statistics for children of target node
        all_stats = collect_action_statistics_mcgs(target_node, c, score_fn, stopping_factor)

        # Calculate children counts
        children_counts = [len(child.children) for child in target_node.children.keys()]

        # Print header
        score_name = score_fn.upper()
        print("\n" + "="*160)
        print(f"CHILDREN OF: {action_str} ({score_name})")
        print("="*160)

        # Tables
        display_statistics_table(all_stats, "Actions by Visits", "visits", score_fn, top, show_node_visits=True)
        display_statistics_table(all_stats, "Actions by Q Value", "q_value", score_fn, top, show_node_visits=True)

        # Summary statistics
        display_summary_statistics(all_stats, children_counts, score_fn, show_node_visits=True)

    def get_visit_stats(self) -> Dict:
        """Get visit statistics for all actions from root."""
        return {merged_action.representative_action: merged_action.total_visits
                for merged_action in self.root.children.values()}

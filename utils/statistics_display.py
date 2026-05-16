"""Shared statistics display utilities for MCGS and MCTS."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from domain.action import Action
from domain.operation import OperatorType


@dataclass
class ActionStatistics:
    """Statistics for a single action."""
    action_str: str
    action_visits: int
    node_visits: int  # Same as action_visits for MCTS, different for MCGS
    q_value: float
    exploration_term: float
    lower_bound: float  # Q - stopping_factor * exploration
    upper_bound: float  # Q + stopping_factor * exploration
    fully_explored: bool
    exploration_progress: str
    action: Action  # For sorting and filtering


def collect_action_statistics_mcgs(
    root_node,
    c: float,
    score_fn: str,
    stopping_factor: float = 1.0
) -> List[ActionStatistics]:
    """Collect action statistics from MCGS root node."""
    stats = []

    for child, merged_action in root_node.children.items():
        action_str = merged_action.get_display_string()
        action_visits = merged_action.total_visits
        node_visits = child.N
        q_value = child.Q
        fully_explored = child.is_fully_explored()

        # Calculate exploration progress
        if child.children:
            visited_children = sum(1 for c in child.children.keys() if c.N > 0)
            total_children = len(child.children)
            exploration_progress = f"{visited_children}/{total_children}"
        else:
            exploration_progress = "0/0"

        # Calculate exploration term based on score function
        if score_fn == "puct":
            if action_visits == 0:
                exploration_term = float('inf')
            else:
                exploration_term = c * merged_action.prior_prob * math.sqrt(root_node.N) / (1 + action_visits)
        elif score_fn == "uct":
            if action_visits == 0:
                exploration_term = float('inf')
            else:
                exploration_term = c * math.sqrt(math.log(root_node.N) / action_visits)
        elif score_fn == "ucd":
            if child.N == 0:
                exploration_term = float('inf')
            else:
                exploration_term = c * math.sqrt(math.log(root_node.N) / child.N)
        else:
            raise ValueError(f"Unknown score function: {score_fn}")

        # Calculate bounds
        if exploration_term != float('inf'):
            lower_bound = q_value - stopping_factor * exploration_term
            upper_bound = q_value + stopping_factor * exploration_term
        else:
            lower_bound = float('-inf')
            upper_bound = float('inf')

        stats.append(ActionStatistics(
            action_str=action_str,
            action_visits=action_visits,
            node_visits=node_visits,
            q_value=q_value,
            exploration_term=exploration_term,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fully_explored=fully_explored,
            exploration_progress=exploration_progress,
            action=merged_action.representative_action
        ))

    return stats


def collect_action_statistics_mcts(
    root_node,
    c: float,
    score_fn: str,
    stopping_factor: float = 1.0
) -> List[ActionStatistics]:
    """Collect action statistics from MCTS root node."""
    stats = []

    for action, child in root_node.children.items():
        action_str = action.get_readable_string()
        visits = child.visit_count
        q_value = child.q_value
        fully_explored = not child.is_leaf() and all(c.visit_count > 0 for c in child.children.values())

        # Calculate exploration progress
        if child.children:
            visited_children = sum(1 for c in child.children.values() if c.visit_count > 0)
            total_children = len(child.children)
            exploration_progress = f"{visited_children}/{total_children}"
        else:
            exploration_progress = "0/0"

        # Calculate exploration term based on score function
        if score_fn == "puct":
            if visits == 0:
                exploration_term = float('inf')
            else:
                exploration_term = c * child.prior_prob * math.sqrt(root_node.visit_count) / (1 + visits)
        elif score_fn == "uct":
            if visits == 0:
                exploration_term = float('inf')
            else:
                exploration_term = c * math.sqrt(2 * math.log(root_node.visit_count) / visits)
        else:
            raise ValueError(f"Unknown score function: {score_fn}")

        # Calculate bounds
        if exploration_term != float('inf'):
            lower_bound = q_value - stopping_factor * exploration_term
            upper_bound = q_value + stopping_factor * exploration_term
        else:
            lower_bound = float('-inf')
            upper_bound = float('inf')

        stats.append(ActionStatistics(
            action_str=action_str,
            action_visits=visits,
            node_visits=visits,  # Same for MCTS
            q_value=q_value,
            exploration_term=exploration_term,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fully_explored=fully_explored,
            exploration_progress=exploration_progress,
            action=action
        ))

    return stats


def display_statistics_table(
    stats: List[ActionStatistics],
    title: str,
    sort_by: str,
    score_fn: str,
    top_n: int = 5,
    show_node_visits: bool = True
) -> None:
    """Display a formatted table of action statistics.

    Args:
        stats: List of ActionStatistics to display
        title: Table title
        sort_by: "visits" or "q_value"
        score_fn: Score function name for display
        top_n: Number of top actions to display
        show_node_visits: Whether to show Node N column (MCGS only)
    """
    if not stats:
        return

    # Sort data
    if sort_by == "visits":
        sorted_stats = sorted(stats, key=lambda x: (x.action_visits, x.action), reverse=True)
    elif sort_by == "q_value":
        sorted_stats = sorted(stats, key=lambda x: (x.q_value, x.action), reverse=True)
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    # Column widths
    action_col_width = 65
    action_visits_col_width = 8
    node_visits_col_width = 8
    fully_explored_col_width = 6
    exploration_col_width = 12
    q_col_width = 10
    explore_term_col_width = 12
    lower_bound_col_width = 12
    upper_bound_col_width = 12
    score_col_width = 12

    score_name = score_fn.upper()

    if show_node_visits:
        total_width = (action_col_width + action_visits_col_width + node_visits_col_width +
                      fully_explored_col_width + exploration_col_width + q_col_width +
                      explore_term_col_width + lower_bound_col_width + upper_bound_col_width + score_col_width)
    else:
        total_width = (action_col_width + action_visits_col_width +
                      fully_explored_col_width + exploration_col_width + q_col_width +
                      explore_term_col_width + lower_bound_col_width + upper_bound_col_width + score_col_width)

    # Print header
    print("\n" + "-"*total_width)
    print(title.upper())
    print("-"*total_width)

    if show_node_visits:
        print(f"{'Action':<{action_col_width}} {'Act N':<{action_visits_col_width}} "
              f"{'Node N':<{node_visits_col_width}} {'Full':<{fully_explored_col_width}} "
              f"{'Exploration':<{exploration_col_width}} {'Q Value':<{q_col_width}} "
              f"{'Explore':<{explore_term_col_width}} {'Lower bound':<{lower_bound_col_width}} "
              f"{'Upper bound':<{upper_bound_col_width}} {f'{score_name} Score':<{score_col_width}}")
    else:
        print(f"{'Action':<{action_col_width}} {'Visits':<{action_visits_col_width}} "
              f"{'Full':<{fully_explored_col_width}} {'Exploration':<{exploration_col_width}} "
              f"{'Q Value':<{q_col_width}} {'Explore':<{explore_term_col_width}} "
              f"{'Lower bound':<{lower_bound_col_width}} {'Upper bound':<{upper_bound_col_width}} "
              f"{f'{score_name} Score':<{score_col_width}}")

    print("-"*total_width)

    # Print data rows
    for stat in sorted_stats[:top_n]:
        q_str = f"{stat.q_value:.3f}"
        explore_str = f"{stat.exploration_term:.3f}" if stat.exploration_term != float('inf') else "inf"
        lower_str = f"{stat.lower_bound:.3f}" if stat.lower_bound != float('-inf') else "-inf"
        upper_str = f"{stat.upper_bound:.3f}" if stat.upper_bound != float('inf') else "inf"
        score = stat.q_value + stat.exploration_term
        score_str = f"{score:.3f}" if stat.exploration_term != float('inf') else "inf"
        full_str = "Yes" if stat.fully_explored else "No"

        if show_node_visits:
            print(f"{stat.action_str:<{action_col_width}} {stat.action_visits:<{action_visits_col_width}} "
                  f"{stat.node_visits:<{node_visits_col_width}} {full_str:<{fully_explored_col_width}} "
                  f"{stat.exploration_progress:<{exploration_col_width}} {q_str:<{q_col_width}} "
                  f"{explore_str:<{explore_term_col_width}} {lower_str:<{lower_bound_col_width}} "
                  f"{upper_str:<{upper_bound_col_width}} {score_str:<{score_col_width}}")
        else:
            print(f"{stat.action_str:<{action_col_width}} {stat.action_visits:<{action_visits_col_width}} "
                  f"{full_str:<{fully_explored_col_width}} {stat.exploration_progress:<{exploration_col_width}} "
                  f"{q_str:<{q_col_width}} {explore_str:<{explore_term_col_width}} "
                  f"{lower_str:<{lower_bound_col_width}} {upper_str:<{upper_bound_col_width}} "
                  f"{score_str:<{score_col_width}}")


def display_summary_statistics(
    stats: List[ActionStatistics],
    children_counts: List[int],
    score_fn: str,
    show_node_visits: bool = True
) -> None:
    """Display summary statistics (min/max/avg) for the statistics table.

    Args:
        stats: List of ActionStatistics
        children_counts: List of children counts for each action
        score_fn: Score function name for display
        show_node_visits: Whether to show Node N column (MCGS only)
    """
    if not stats:
        return

    # Column widths
    action_col_width = 65
    action_visits_col_width = 8
    node_visits_col_width = 8
    fully_explored_col_width = 6
    exploration_col_width = 12
    q_col_width = 10
    explore_term_col_width = 12
    lower_bound_col_width = 12
    upper_bound_col_width = 12
    score_col_width = 12

    score_name = score_fn.upper()

    if show_node_visits:
        total_width = (action_col_width + action_visits_col_width + node_visits_col_width +
                      fully_explored_col_width + exploration_col_width + q_col_width +
                      explore_term_col_width + lower_bound_col_width + upper_bound_col_width + score_col_width)
    else:
        total_width = (action_col_width + action_visits_col_width +
                      fully_explored_col_width + exploration_col_width + q_col_width +
                      explore_term_col_width + lower_bound_col_width + upper_bound_col_width + score_col_width)

    # Extract values
    action_visits_counts = [s.action_visits for s in stats]
    node_visits_counts = [s.node_visits for s in stats]
    q_values = [s.q_value for s in stats]
    exploration_terms = [s.exploration_term for s in stats if s.exploration_term != float('inf')]
    lower_bounds = [s.lower_bound for s in stats if s.lower_bound != float('-inf')]
    upper_bounds = [s.upper_bound for s in stats if s.upper_bound != float('inf')]
    scores = [s.q_value + s.exploration_term for s in stats if s.exploration_term != float('inf')]
    fully_explored_flags = [s.fully_explored for s in stats]

    print("-"*total_width)

    # Header
    if show_node_visits:
        print(f"{'STATISTICS':<{action_col_width}} {'Act N':<{action_visits_col_width}} "
              f"{'Node N':<{node_visits_col_width}} {'Full':<{fully_explored_col_width}} "
              f"{'Exploration':<{exploration_col_width}} {'Q Value':<{q_col_width}} "
              f"{'Explore':<{explore_term_col_width}} {'Lower bound':<{lower_bound_col_width}} "
              f"{'Upper bound':<{upper_bound_col_width}} {f'{score_name} Score':<{score_col_width}}")
    else:
        print(f"{'STATISTICS':<{action_col_width}} {'Visits':<{action_visits_col_width}} "
              f"{'Full':<{fully_explored_col_width}} {'Exploration':<{exploration_col_width}} "
              f"{'Q Value':<{q_col_width}} {'Explore':<{explore_term_col_width}} "
              f"{'Lower bound':<{lower_bound_col_width}} {'Upper bound':<{upper_bound_col_width}} "
              f"{f'{score_name} Score':<{score_col_width}}")

    print("-"*total_width)

    # Max values
    max_action_visits = max(action_visits_counts) if action_visits_counts else 0
    max_node_visits = max(node_visits_counts) if node_visits_counts else 0
    max_q = max(q_values) if q_values else 0
    max_explore = max(exploration_terms) if exploration_terms else 0
    max_lower = max(lower_bounds) if lower_bounds else 0
    max_upper = max(upper_bounds) if upper_bounds else 0
    max_score = max(scores) if scores else 0
    max_children = max(children_counts) if children_counts else 0
    fully_explored_count = sum(fully_explored_flags)

    if show_node_visits:
        print(f"{'Max':<{action_col_width}} {max_action_visits:<{action_visits_col_width}} "
              f"{max_node_visits:<{node_visits_col_width}} {fully_explored_count:<{fully_explored_col_width}} "
              f"{max_children:<{exploration_col_width}} {max_q:<{q_col_width}.3f} "
              f"{max_explore:<{explore_term_col_width}.3f} {max_lower:<{lower_bound_col_width}.3f} "
              f"{max_upper:<{upper_bound_col_width}.3f} {max_score:<{score_col_width}.3f}")
    else:
        print(f"{'Max':<{action_col_width}} {max_action_visits:<{action_visits_col_width}} "
              f"{fully_explored_count:<{fully_explored_col_width}} {max_children:<{exploration_col_width}} "
              f"{max_q:<{q_col_width}.3f} {max_explore:<{explore_term_col_width}.3f} "
              f"{max_lower:<{lower_bound_col_width}.3f} {max_upper:<{upper_bound_col_width}.3f} "
              f"{max_score:<{score_col_width}.3f}")

    # Min values
    min_action_visits = min(action_visits_counts) if action_visits_counts else 0
    min_node_visits = min(node_visits_counts) if node_visits_counts else 0
    min_q = min(q_values) if q_values else 0
    min_explore = min(exploration_terms) if exploration_terms else 0
    min_lower = min(lower_bounds) if lower_bounds else 0
    min_upper = min(upper_bounds) if upper_bounds else 0
    min_score = min(scores) if scores else 0
    min_children = min(children_counts) if children_counts else 0

    if show_node_visits:
        print(f"{'Min':<{action_col_width}} {min_action_visits:<{action_visits_col_width}} "
              f"{min_node_visits:<{node_visits_col_width}} {0:<{fully_explored_col_width}} "
              f"{min_children:<{exploration_col_width}} {min_q:<{q_col_width}.3f} "
              f"{min_explore:<{explore_term_col_width}.3f} {min_lower:<{lower_bound_col_width}.3f} "
              f"{min_upper:<{upper_bound_col_width}.3f} {min_score:<{score_col_width}.3f}")
    else:
        print(f"{'Min':<{action_col_width}} {min_action_visits:<{action_visits_col_width}} "
              f"{0:<{fully_explored_col_width}} {min_children:<{exploration_col_width}} "
              f"{min_q:<{q_col_width}.3f} {min_explore:<{explore_term_col_width}.3f} "
              f"{min_lower:<{lower_bound_col_width}.3f} {min_upper:<{upper_bound_col_width}.3f} "
              f"{min_score:<{score_col_width}.3f}")

    # Average values
    avg_action_visits = sum(action_visits_counts) / len(action_visits_counts) if action_visits_counts else 0
    avg_node_visits = sum(node_visits_counts) / len(node_visits_counts) if node_visits_counts else 0
    avg_q = sum(q_values) / len(q_values) if q_values else 0
    avg_explore = sum(exploration_terms) / len(exploration_terms) if exploration_terms else 0
    avg_lower = sum(lower_bounds) / len(lower_bounds) if lower_bounds else 0
    avg_upper = sum(upper_bounds) / len(upper_bounds) if upper_bounds else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_children = sum(children_counts) / len(children_counts) if children_counts else 0
    fully_explored_ratio = fully_explored_count / len(fully_explored_flags) if fully_explored_flags else 0

    if show_node_visits:
        print(f"{'Average':<{action_col_width}} {avg_action_visits:<{action_visits_col_width}.1f} "
              f"{avg_node_visits:<{node_visits_col_width}.1f} {fully_explored_ratio:<{fully_explored_col_width}.2f} "
              f"{avg_children:<{exploration_col_width}.1f} {avg_q:<{q_col_width}.3f} "
              f"{avg_explore:<{explore_term_col_width}.3f} {avg_lower:<{lower_bound_col_width}.3f} "
              f"{avg_upper:<{upper_bound_col_width}.3f} {avg_score:<{score_col_width}.3f}")
    else:
        print(f"{'Average':<{action_col_width}} {avg_action_visits:<{action_visits_col_width}.1f} "
              f"{fully_explored_ratio:<{fully_explored_col_width}.2f} {avg_children:<{exploration_col_width}.1f} "
              f"{avg_q:<{q_col_width}.3f} {avg_explore:<{explore_term_col_width}.3f} "
              f"{avg_lower:<{lower_bound_col_width}.3f} {avg_upper:<{upper_bound_col_width}.3f} "
              f"{avg_score:<{score_col_width}.3f}")

    print("="*total_width)


def filter_by_operators(
    stats: List[ActionStatistics],
    focus_operators: List[OperatorType]
) -> List[ActionStatistics]:
    """Filter statistics by operator types."""
    return [s for s in stats if s.action.operator in focus_operators]

"""Algorithm domain model."""

from dataclasses import dataclass, field
from typing import Dict, List
from copy import deepcopy

from domain.operation import Operation
from domain.action import Action, LoopType, ActionType

@dataclass
class Algorithm:
    """Represents a linear solver algorithm as a sequence of operations."""

    setup_loop: List[Operation] = field(default_factory=list)
    forward_loop: List[Operation] = field(default_factory=list)
    update_loop: List[Operation] = field(default_factory=list)

    def apply_action(self, action: Action):
        """Apply an action to the algorithm."""
        if action.loop_select == LoopType.SETUP:
            if action.action_type == ActionType.INSERT:
                self.setup_loop.insert(action.position, action.operation)
            elif action.action_type == ActionType.DELETE:
                self.remove_operation(action.loop_select, action.position)
        elif action.loop_select == LoopType.FORWARD:
            if action.action_type == ActionType.INSERT:
                self.forward_loop.insert(action.position, action.operation)
            elif action.action_type == ActionType.DELETE:
                self.remove_operation(action.loop_select, action.position)
        elif action.loop_select == LoopType.UPDATE:
            if action.action_type == ActionType.INSERT:
                self.update_loop.insert(action.position, action.operation)
            elif action.action_type == ActionType.DELETE:
                self.remove_operation(action.loop_select, action.position)
    
    def remove_operation(self, loop_type: LoopType, position: int):
        """Remove an operation from the algorithm."""
        if loop_type == LoopType.SETUP:
            if 0 <= position < len(self.setup_loop):
                self.setup_loop.pop(position)
            else:
                raise ValueError(f"Invalid position: {position} for setup loop")
        elif loop_type == LoopType.FORWARD:
            if 0 <= position < len(self.forward_loop):
                self.forward_loop.pop(position)
            else:
                raise ValueError(f"Invalid position: {position} for forward loop")
        elif loop_type == LoopType.UPDATE:
            if 0 <= position < len(self.update_loop):
                self.update_loop.pop(position)
            else:
                raise ValueError(f"Invalid position: {position} for update loop")

    def get_total_operations(self) -> int:
        """Get total number of operations in the algorithm."""
        return len(self.setup_loop) + len(self.forward_loop) + len(self.update_loop)

    def clone(self) -> 'Algorithm':
        """Create a deep copy of the algorithm."""
        return Algorithm(
            setup_loop=deepcopy(self.setup_loop),
            forward_loop=deepcopy(self.forward_loop),
            update_loop=deepcopy(self.update_loop)
        )

    def is_empty(self) -> bool:
        """Check if algorithm has no operations."""
        return len(self.setup_loop) == 0 and len(self.forward_loop) == 0 and len(self.update_loop) == 0

    def get_readable_representation(self) -> str:
        """Get human-readable string representation."""
        lines = []
        lines.append("--------------------")

        # Setup loop
        for op in self.setup_loop:
            lines.append(op.get_readable_string())

        lines.append("--------------------")
        lines.append("==for t in range(0, T)==")
        lines.append("--------------------")

        # Forward loop
        for op in self.forward_loop:
            lines.append(f"    {op.get_readable_string()}")

        lines.append("    ----")

        # Update loop
        for op in self.update_loop:
            lines.append(f"    {op.get_readable_string()}")

        lines.append("--------------------")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return f"Algorithm(setup={len(self.setup_loop)}, forward={len(self.forward_loop)}, update={len(self.update_loop)})"

    def __eq__(self, other) -> bool:
        """Check equality with another algorithm."""
        if not isinstance(other, Algorithm):
            return False
        return (self.setup_loop == other.setup_loop and
                self.forward_loop == other.forward_loop and
                self.update_loop == other.update_loop)

    def __hash__(self) -> int:
        """Hash for use in sets/dictionaries."""
        setup_tuple = tuple(tuple(op.to_list()) for op in self.setup_loop)
        forward_tuple = tuple(tuple(op.to_list()) for op in self.forward_loop)
        update_tuple = tuple(tuple(op.to_list()) for op in self.update_loop)
        return hash((setup_tuple, forward_tuple, update_tuple))

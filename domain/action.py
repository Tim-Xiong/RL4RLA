"""Action and action type definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Union

from domain.operation import Operation, OperatorType, OperandType


class LoopType(Enum):
    """Loop type enum."""
    SETUP = 0
    FORWARD = 1
    UPDATE = 2

    def __repr__(self):
        return f"LoopType.{self.name}"
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class ActionType(Enum):
    """Action type enum."""
    INSERT = 0
    DELETE = 1

    def __repr__(self):
        return f"ActionType.{self.name}"
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

@dataclass(frozen=True, order=True)
class Action:
    """Represents an action in the MCTS search (operation + location info)."""

    operation: Operation
    loop_select: LoopType  # 0 for setup, 1 for forward, 2 for update
    action_type: ActionType  # 0 for insert, 1 for delete
    position: int  # Position for insertion and deletion actions (required)

    @property
    def operator(self) -> OperatorType:
        return self.operation.operator
    
    @property
    def operand1(self) -> OperandType:
        return self.operation.operand1
    
    @property
    def operand2(self) -> OperandType:
        return self.operation.operand2
    
    @property
    def target(self) -> OperandType:
        return self.operation.target

    def to_list(self) -> List[Union[OperatorType, OperandType, LoopType, ActionType, int]]:
        """Convert to list format"""
        op_list = self.operation.to_list()
        action_parts = op_list + [self.loop_select, self.action_type, self.position]
        return action_parts

    @classmethod
    def from_list(cls, action_list: List) -> 'Action':
        """Create action from list format."""
        if len(action_list) < 7:
            raise ValueError(f"Action list must have at least 7 parts: {action_list}")

        operation = Operation.from_list(action_list[:4])
        loop_select = LoopType(action_list[4])
        action_type = ActionType(action_list[5])
        position = action_list[6]

        return cls(operation, loop_select, action_type, position)

    def get_readable_string(self) -> str:
        """Get human-readable string representation of the action."""
        op_str = self.operation.get_readable_string()
        return f"{op_str}, {self.loop_select}, {self.action_type}, {self.position}"

    def to_dict(self) -> dict:
        """Convert action to dictionary for JSON serialization."""
        return {
            "operation": self.operation.to_dict(),
            "loop_select": self.loop_select.name,
            "action_type": self.action_type.name,
            "position": self.position
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Action':
        """Create action from dictionary."""
        return cls(
            operation=Operation.from_dict(data["operation"]),
            loop_select=LoopType[data["loop_select"]],
            action_type=ActionType[data["action_type"]],
            position=data["position"]
        )

"""Operation and operand definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Union


class OperatorType(Enum):
    """Mathematical operators available in the system."""

    # Vector operations
    VEC_VEC_ADD = (0, "vec-vec-add", "+")
    VEC_VEC_SUB = (1, "vec-vec-sub", "-")
    VEC_VEC_DOT = (2, "vec-vec-innerprod", "dot")

    # Matrix-vector operations
    MAT_VEC_MUL = (3, "Mat-vec-mul", "@")
    VEC_MAT_MUL = (4, "vec-mat-mul", "@")

    # Scalar operations
    SCALAR_VEC_MUL = (5, "scalar-vec-mul", "*")
    SCALAR_DIV = (6, "scalar-div", "/")

    # Matrix operations
    SKETCH = (7, "sketch", "sketch")
    MAT_MAT_MUL = (8, "Mat-mat-mul", "@")
    HHQR = (9, "HHQR", "HHQR")
    TRIANGULAR_SOLVE = (10, "triangular-solve", "\\")
    MAT_INV = (11, "Mat-inv", "^-1")
    LSQR = (12, "lsqr", "lsqr")
    MAT_MAT_TRANS_MUL = (13, "Mat-mat-trans-mul", "@ trans(")
    MAT_TRANS_MAT_MUL = (14, "Mat-trans-mat-mul", "@")

    # Special operations
    LEVERAGE_SCORE = (15, "compute-leverage-score", "leverage_score")
    SUBSAMPLING = (16, "subsampling", "subsampling")

    # Control operations
    DELETE = (100, "delete", "delete")
    DO_NOTHING = (-1, "do_nothing", "do_nothing")

    def __init__(self, id_val: int, name: str, symbol: str):
        self.id = id_val
        self.operation_name = name
        self.symbol = symbol
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value[0] < other.value[0]
        return NotImplemented
    
    def __repr__(self):
        return f"OperatorType.{self.name}"

class OperandType(Enum):
    """Available operands in the system."""

    # Vector operands
    V1 = (0, "v1")
    V2 = (1, "v2")
    V3 = (2, "v3") # reserved for leverage score subsampling

    # Scalar operands
    A1 = (3, "a1")
    A2 = (4, "a2")

    # System operands
    B = (5, "b")
    A = (6, "A")
    LR = (7, "lr")
    X_T = (8, "x_t")
    UPDATE = (9, "update")

    # Matrix operands
    R1 = (10, "R1")
    R2 = (11, "R2")
    R3 = (12, "R3") # reserved for preconditioner

    # Special operand
    NONE = (-1, "-")

    def __init__(self, id_val: int, symbol: str):
        self.id = id_val
        self.symbol = symbol
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value[0] < other.value[0]  # Compare by ID
        return NotImplemented
    
    def __repr__(self):
        return f"OperandType.{self.name}"


@dataclass(frozen=True, order=True)
class Operation:
    """Represents a single mathematical operation."""

    operator: OperatorType
    operand1: OperandType
    operand2: OperandType
    target: OperandType

    def to_list(self) -> List[Union[OperatorType, OperandType]]:
        """Convert to list format used in original code."""
        return [
            self.operator,
            self.operand1,
            self.operand2,
            self.target
        ]

    @classmethod
    def from_list(cls, op_list: List[Union[OperatorType, OperandType]]) -> 'Operation':
        """Create operation from list format."""
        if len(op_list) < 4:
            raise ValueError(f"Operation list must have at least 4 elements: {op_list}")

        operator = op_list[0]
        operand1 = op_list[1]
        operand2 = op_list[2]
        target = op_list[3]

        if not isinstance(operator, OperatorType):
            raise ValueError(f"Invalid operator type: {operator}")
        if not isinstance(operand1, OperandType):
            raise ValueError(f"Invalid operand1 type: {operand1}")
        if not isinstance(operand2, OperandType):
            raise ValueError(f"Invalid operand2 type: {operand2}")
        if not isinstance(target, OperandType):
            raise ValueError(f"Invalid target type: {target}")

        return cls(operator, operand1, operand2, target)

    def get_readable_string(self) -> str:
        """Get human-readable representation."""
        if self.operator == OperatorType.DO_NOTHING:
            return "do nothing"
        elif self.operator == OperatorType.DELETE:
            return "delete"
        elif self.operator == OperatorType.VEC_MAT_MUL:
            return f"{self.target.name} = {self.operand2.name}.T {self.operator.symbol} {self.operand1.name}"
        elif self.operator == OperatorType.MAT_TRANS_MAT_MUL:
            return f"{self.target.name} = {self.operand1.name}.T {self.operator.symbol} {self.operand2.name}"
        elif self.operator == OperatorType.SUBSAMPLING:
            return f"{self.target.name} = {self.operator.symbol} {self.operand1.name}"
        else:
            return f"{self.target.name} = {self.operand1.name} {self.operator.symbol} {self.operand2.name}"

    def to_dict(self) -> dict:
        """Convert operation to dictionary for JSON serialization."""
        return {
            "operator": self.operator.name,
            "operand1": self.operand1.name,
            "operand2": self.operand2.name,
            "target": self.target.name
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Operation':
        """Create operation from dictionary."""
        return cls(
            operator=OperatorType[data["operator"]],
            operand1=OperandType[data["operand1"]],
            operand2=OperandType[data["operand2"]],
            target=OperandType[data["target"]]
        )

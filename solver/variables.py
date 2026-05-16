"""Variable management system for algorithm execution.

This module implements a register-based variable management system that tracks:
- Variable initialization state
- Mathematical type compatibility
- Availability for use as operands or targets
- Position-dependent variable availability (insert only)
"""

import json

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from domain.operation import OperandType, Operation, OperatorType
from domain.action import ActionType, LoopType, Action
from domain.algorithm import Algorithm


class VariableType(Enum):
    """Types of variables in the register system."""
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    NONE = "none"


class VariableCategory(Enum):
    """Categories of variables based on their role."""
    CACHE = "cache"      # Temporary storage registers (a1, a2, v1, v2, R1, R2)
    SYSTEM = "system"    # Read-only system variables (A, b, x_t, update, lr)
    RESERVED = "reserved" # Reserved variables (v3, NONE)

@dataclass
class VariableInitialization:
    """Tracks where and when a variable was initialized."""
    loop_type: LoopType
    position: int  # Position in the specific loop where variable was initialized


@dataclass
class VariableInfo:
    """Information about a variable in the system."""
    var_type: VariableType
    category: VariableCategory
    operand_type: OperandType
    target_allowed_loops: List[LoopType] = field(
        default_factory=lambda: [LoopType.SETUP, LoopType.FORWARD, LoopType.UPDATE])
    operand_allowed_loops: List[LoopType] = field(
        default_factory=lambda: [LoopType.SETUP, LoopType.FORWARD, LoopType.UPDATE])
    initialization: Optional[VariableInitialization] = None


@dataclass
class AlgorithmState:
    """Tracks the state of variables and algorithm during execution.
    
    This class is used by MCTSNode to track variable availability and
    can be used for caching execution results and rewards.
    """

    algorithm: Algorithm = field(default_factory=Algorithm)
    variables: Dict[OperandType, VariableInfo] = field(default_factory=dict)
    step: int = field(default=0)

    def __post_init__(self):
        if not self.variables:
            self._initialize_variables()

    @property
    def setup_loop_length(self) -> int:
        """Get current length of setup loop."""
        return len(self.algorithm.setup_loop)

    @property
    def forward_loop_length(self) -> int:
        """Get current length of forward loop."""
        return len(self.algorithm.forward_loop)

    @property
    def update_loop_length(self) -> int:
        """Get current length of update loop."""
        return len(self.algorithm.update_loop)

    def _initialize_variables(self):
        """Initialize all variables with their metadata."""
        # Scalar cache variables
        self.variables[OperandType.A1] = VariableInfo(
            VariableType.SCALAR, VariableCategory.CACHE, OperandType.A1
        )
        self.variables[OperandType.A2] = VariableInfo(
            VariableType.SCALAR, VariableCategory.CACHE, OperandType.A2
        )

        # Vector cache variables
        self.variables[OperandType.V1] = VariableInfo(
            VariableType.VECTOR, VariableCategory.CACHE, OperandType.V1,
            target_allowed_loops=[LoopType.FORWARD, LoopType.UPDATE],
        )
        self.variables[OperandType.V2] = VariableInfo(
            VariableType.VECTOR, VariableCategory.CACHE, OperandType.V2,
        )

        # Matrix cache variables
        self.variables[OperandType.R1] = VariableInfo(
            VariableType.MATRIX, VariableCategory.CACHE, OperandType.R1,
            # target_allowed_loops=[LoopType.SETUP],
        )
        self.variables[OperandType.R2] = VariableInfo(
            VariableType.MATRIX, VariableCategory.CACHE, OperandType.R2,
        )

        # System variables (read-only)
        self.variables[OperandType.A] = VariableInfo(
            VariableType.MATRIX, VariableCategory.SYSTEM, OperandType.A
        )
        self.variables[OperandType.B] = VariableInfo(
            VariableType.VECTOR, VariableCategory.SYSTEM, OperandType.B
        )
        self.variables[OperandType.X_T] = VariableInfo(
            VariableType.VECTOR, VariableCategory.SYSTEM, OperandType.X_T
        )
        self.variables[OperandType.UPDATE] = VariableInfo(
            VariableType.VECTOR, VariableCategory.SYSTEM, OperandType.UPDATE
        )
        self.variables[OperandType.LR] = VariableInfo(
            VariableType.SCALAR, VariableCategory.SYSTEM, OperandType.LR
        )

        # Special handling for V3 and NONE
        self.variables[OperandType.V3] = VariableInfo(
            VariableType.VECTOR, VariableCategory.RESERVED, OperandType.V3
        )
        self.variables[OperandType.NONE] = VariableInfo(
            VariableType.NONE, VariableCategory.RESERVED, OperandType.NONE
        )

    def to_dict(self) -> Dict[str, List[Operation]]:
        """Convert algorithm state to dictionary."""
        return {
            "setup_loop": [op.to_list() for op in self.algorithm.setup_loop],
            "forward_loop": [op.to_list() for op in self.algorithm.forward_loop],
            "update_loop": [op.to_list() for op in self.algorithm.update_loop]
        }
    
    @classmethod
    def from_json(cls, json_data: str) -> 'AlgorithmState':
        """Create an algorithm state from a dictionary."""
        op_history = json.loads(json_data)
        algorithm = AlgorithmState()
        manager = AlgorithmStateManager()
        for i, op in enumerate(op_history.get("setup_loop", [])):
            algorithm = manager.apply_action_to_state(algorithm, Action.from_list(
                [OperatorType[op[0]], OperandType[op[1]], OperandType[op[2]], OperandType[op[3]]]
                + [LoopType.SETUP, ActionType.INSERT, i]
            ))
        for i, op in enumerate(op_history.get("forward_loop", [])):
            algorithm = manager.apply_action_to_state(algorithm, Action.from_list(
                [OperatorType[op[0]], OperandType[op[1]], OperandType[op[2]], OperandType[op[3]]]
                + [LoopType.FORWARD, ActionType.INSERT, i]
            ))
        for i, op in enumerate(op_history.get("update_loop", [])):
            algorithm = manager.apply_action_to_state(algorithm, Action.from_list(
                [OperatorType[op[0]], OperandType[op[1]], OperandType[op[2]], OperandType[op[3]]]
                + [LoopType.UPDATE, ActionType.INSERT, i]
            ))
        return algorithm

    @classmethod
    def from_dict(cls, op_history: Dict[str, List[Operation]]) -> 'AlgorithmState':
        """Create an algorithm state from a dictionary."""
        algorithm = AlgorithmState()
        manager = AlgorithmStateManager()
        for i, op in enumerate(op_history.get("setup_loop", [])):
            algorithm = manager.apply_action_to_state(algorithm, Action.from_list(
                op + [LoopType.SETUP, ActionType.INSERT, i]
            ))
        for i, op in enumerate(op_history.get("forward_loop", [])):
            algorithm = manager.apply_action_to_state(algorithm, Action.from_list(
                op + [LoopType.FORWARD, ActionType.INSERT, i]
            ))
        for i, op in enumerate(op_history.get("update_loop", [])):
            algorithm = manager.apply_action_to_state(algorithm, Action.from_list(
                op + [LoopType.UPDATE, ActionType.INSERT, i]
            ))
        return algorithm

    def clone(self) -> 'AlgorithmState':
        """Create a deep copy of the algorithm state."""
        return AlgorithmState(
            algorithm=self.algorithm.clone(),
            variables=deepcopy(self.variables),
            step=self.step
        )

    def mark_initialized(self, operand: OperandType, loop_type: LoopType, position: int):
        """Mark a variable as initialized at a specific position."""
        if operand not in self.variables:
            return

        variable = self.variables[operand]

        if variable.category == VariableCategory.CACHE:
            self.variables[operand].initialization = VariableInitialization(loop_type, position)
        elif variable.category == VariableCategory.SYSTEM:
            # System variables are always initialized
            return
        elif variable.category == VariableCategory.RESERVED:
            if operand == OperandType.V3:
                self.variables[operand].initialization = VariableInitialization(loop_type, position)
            elif operand == OperandType.NONE:
                return

    def can_use_as_operand(self, operand: OperandType, loop_type: LoopType, position: int) -> bool:
        """Check if a variable can be used as an operand considering position."""
        if operand not in self.variables:
            return False

        variable = self.variables[operand]

        if loop_type not in variable.operand_allowed_loops:
            return False

        if variable.category == VariableCategory.CACHE:
            return self._is_variable_initialized(operand, loop_type, position)

        elif variable.category == VariableCategory.SYSTEM:
            # System variables are always available
            return True

        elif variable.category == VariableCategory.RESERVED:
            if operand == OperandType.V3:
                return True
            elif operand == OperandType.NONE:
                return True

        return False

    def can_use_as_target(self, operand: OperandType, loop_type: LoopType, position: int) -> bool:
        """Check if a variable can be used as a target considering position."""
        if operand not in self.variables:
            return False

        variable = self.variables[operand]

        if loop_type not in variable.target_allowed_loops:
            return False

        if variable.category == VariableCategory.CACHE:
            if self._is_variable_initialized(operand, loop_type, position):
                return True
            uninitialized_of_type = []
            for other_operand, other_variable in self.variables.items():
                if (variable.var_type == other_variable.var_type and
                    not self._is_variable_initialized(other_operand, loop_type, position)):
                    uninitialized_of_type.append(other_operand)

            if len(uninitialized_of_type) == 1:
                return True

            if len(uninitialized_of_type) > 1:
                smallest_enum = min(uninitialized_of_type, key=lambda x: x.value)
                return operand == smallest_enum

        elif variable.category == VariableCategory.SYSTEM:
            # System variables are read only
            return False

        elif variable.category == VariableCategory.RESERVED:
            if operand == OperandType.V3:
                return True
            elif operand == OperandType.NONE:
                return True

        return False

    def increment_step(self):
        """Increment the step count."""
        self.step += 1

    def get_variable_type(self, operand: OperandType) -> Optional[VariableType]:
        """Get the mathematical type of a variable."""
        if operand in self.variables:
            return self.variables[operand].var_type
        return None

    def get_initialized_variables(self) -> List[OperandType]:
        """Get all initialized variables."""
        initialized_variables = {}
        for operand, variable in self.variables.items():
            if variable.initialization is not None:
                initialized_variables[operand] = variable.initialization
        return initialized_variables

    def _is_variable_initialized(self, operand: OperandType, loop_type: LoopType,
                                 position: int) -> bool:
        """Check if a variable is initialized at a specific position."""
        if operand not in self.variables:
            return False

        variable = self.variables[operand]
        init_info = variable.initialization

        if variable.category == VariableCategory.CACHE:
            # Cache variables state depends on init position
            if init_info is None:
                return False

            # Check position to determine if variable is initialized
            if init_info.loop_type == loop_type:
                return init_info.position < position
            else:
                # Loop execution order: SETUP → FORWARD → UPDATE
                loop_order = {LoopType.SETUP: 0, LoopType.FORWARD: 1, LoopType.UPDATE: 2}
                return loop_order[init_info.loop_type] < loop_order[loop_type]

        elif variable.category == VariableCategory.SYSTEM:
            # System variables are always initialized
            return True

        elif variable.category == VariableCategory.RESERVED:
            if operand == OperandType.V3:  # V3 has default value
                return True
            elif operand == OperandType.NONE:
                return True

        return False

    def _shift_variable_positions(self, loop_type: LoopType, from_position: int, shift_amount: int):
        """Shift variable initialization positions when operations are inserted."""
        for variable in self.variables.values():
            if (variable.initialization and
                variable.initialization.loop_type == loop_type and
                variable.initialization.position >= from_position):
                variable.initialization.position += shift_amount

    def __eq__(self, other) -> bool:
        """Check equality with another AlgorithmState."""
        if not isinstance(other, AlgorithmState):
            return False

        return self.algorithm == other.algorithm and self.step == other.step

    def __hash__(self) -> int:
        """Hash for use in caching."""
        return hash((self.algorithm, self.step))


class AlgorithmStateManager:
    """Manages algorithm state transitions and variable tracking.
    
    Focused responsibilities:
    - Apply actions to algorithm states
    - Track variable initialization and availability
    - Manage redundancy removal
    - Handle state cloning and updates
    
    Note: Operation constraints and type rules are handled by ActionGenerator.
    """

    def apply_action_to_state(self, state: AlgorithmState, action: Action) -> AlgorithmState:
        """Apply an action to a state, returning new state with updated algorithm and variable availability."""
        new_state = state.clone()

        # Increment step count
        new_state.increment_step()

        # Apply action to algorithm
        new_state.algorithm.apply_action(action)

        # For INSERT actions, the position is explicitly specified
        target_pos = action.position

        # Shift existing variable positions after insertion point
        if action.action_type == ActionType.INSERT:
            new_state._shift_variable_positions(action.loop_select, action.position, 1)

        var_info = new_state.variables[action.operation.target]
        # Mark target as initialized at the determined position if target will be overwritten
        if (var_info.category == VariableCategory.CACHE and
            action.operation.operand1 != action.operation.target and
            action.operation.operand2 != action.operation.target):
            new_state.mark_initialized(action.operation.target, action.loop_select, target_pos)
            self._remove_redundancies_iteratively(new_state)

        return new_state

    def _remove_redundancies_iteratively(self, state: AlgorithmState):
        """
        Iteratively detect and remove redundant operations until no more are found.
        This handles cascading redundancies that may appear after removing operations.
        """
        while True:
            redundant_operations = self._find_all_redundant_operations(state)
            if not redundant_operations:
                break  # No more redundancies found
                
            # Remove redundant operations (process in reverse order to maintain positions)
            # We remove the LOWER position operations (the redundant initializations)
            # and keep the HIGHER position operations (the overwrites)
            for loop_type, position in sorted(redundant_operations, key=lambda x: (x[0].value, x[1]), reverse=True):
                self._remove_operation_at_position(state, loop_type, position)

    def _find_all_redundant_operations(self, state: AlgorithmState) -> List[Tuple[LoopType, int]]:
        """
        Find all redundant operations in the current algorithm state.
        Returns list of (loop_type, position) tuples for redundant operations.
        
        Note: This identifies LOWER position operations that are redundant because
        they initialize variables that get overwritten at HIGHER positions without use.
        """
        redundant_operations = []
        
        # Check setup loop operations
        for pos, operation in enumerate(state.algorithm.setup_loop):
            if self._is_operation_redundant(state, operation.target, LoopType.SETUP, pos):
                redundant_operations.append((LoopType.SETUP, pos))
        
        # Check forward loop operations
        for pos, operation in enumerate(state.algorithm.forward_loop):
            if self._is_operation_redundant(state, operation.target, LoopType.FORWARD, pos):
                redundant_operations.append((LoopType.FORWARD, pos))
        
        # Check update loop operations
        for pos, operation in enumerate(state.algorithm.update_loop):
            if self._is_operation_redundant(state, operation.target, LoopType.UPDATE, pos):
                redundant_operations.append((LoopType.UPDATE, pos))
                
        return redundant_operations

    def _is_operation_redundant(self, state: AlgorithmState, target: OperandType,
                               loop_type: LoopType, position: int) -> bool:
        """
        Check if an operation at the given position is redundant.
        An operation is redundant if:
        1. It initializes a cache variable at position (lower position)
        2. The same variable is overwritten later at a higher position
        3. The variable is never used as operand between these two positions
        """
        if target not in state.variables:
            return False
            
        var_info = state.variables[target]
        
        # Only cache variables can be redundant
        if var_info.category != VariableCategory.CACHE:
            return False
            
        # Find if this variable is overwritten later (at higher positions)
        overwrite_position = self._find_next_overwrite_position(state, target, loop_type, position)
        
        if overwrite_position is None:
            # Variable is not overwritten later, so this operation is not redundant
            return False
            
        # Check if variable is used as operand between current position and overwrite position
        is_used_between = self._is_variable_used_between_positions(
            state, target, loop_type, position, overwrite_position[0], overwrite_position[1]
        )
        
        # Operation is redundant if variable is overwritten later without being used in between
        return not is_used_between

    def _find_next_overwrite_position(self, state: AlgorithmState, var: OperandType,
                                     loop_type: LoopType, position: int) -> Optional[Tuple[LoopType, int]]:
        """
        Find the next position where this variable is overwritten (used as target).
        Returns (loop_type, position) tuple if found, None otherwise.
        """
        # Check remaining operations in the same loop
        if loop_type == LoopType.SETUP:
            loop_ops = state.algorithm.setup_loop
        elif loop_type == LoopType.FORWARD:
            loop_ops = state.algorithm.forward_loop
        else:
            loop_ops = state.algorithm.update_loop
            
        # Check operations after current position in the same loop
        for i in range(position + 1, len(loop_ops)):
            operation = loop_ops[i]
            if operation.target == var:
                return (loop_type, i)
                
        # Check if overwritten in later loops (following execution order)
        if loop_type == LoopType.SETUP:
            # Check forward loop
            for i, operation in enumerate(state.algorithm.forward_loop):
                if operation.target == var:
                    return (LoopType.FORWARD, i)
            # Check update loop
            for i, operation in enumerate(state.algorithm.update_loop):
                if operation.target == var:
                    return (LoopType.UPDATE, i)
        elif loop_type == LoopType.FORWARD:
            # Check update loop
            for i, operation in enumerate(state.algorithm.update_loop):
                if operation.target == var:
                    return (LoopType.UPDATE, i)
                    
        return None

    def _is_variable_used_between_positions(self, state: AlgorithmState, var: OperandType,
                                          start_loop: LoopType, start_pos: int,
                                          end_loop: LoopType, end_pos: int) -> bool:
        """
        Check if a variable is used as operand between two positions.
        """
        if start_loop == end_loop:
            # Same loop: check operations between start_pos+1 and end_pos (inclusive)
            if start_loop == LoopType.SETUP:
                loop_ops = state.algorithm.setup_loop
            elif start_loop == LoopType.FORWARD:
                loop_ops = state.algorithm.forward_loop
            else:
                loop_ops = state.algorithm.update_loop
                
            for i in range(start_pos + 1, end_pos + 1):
                operation = loop_ops[i]
                if operation.operand1 == var or operation.operand2 == var:
                    return True
        else:
            # Cross-loop: check all loops between start_loop and end_loop in execution order
            loop_order = [LoopType.SETUP, LoopType.FORWARD, LoopType.UPDATE]
            start_idx = loop_order.index(start_loop)
            end_idx = loop_order.index(end_loop)
            
            # Check remaining operations in start loop after start_pos
            if start_loop == LoopType.SETUP:
                start_loop_ops = state.algorithm.setup_loop
            elif start_loop == LoopType.FORWARD:
                start_loop_ops = state.algorithm.forward_loop
            else:
                start_loop_ops = state.algorithm.update_loop
                
            for i in range(start_pos + 1, len(start_loop_ops)):
                operation = start_loop_ops[i]
                if operation.operand1 == var or operation.operand2 == var:
                    return True
            
            # Check all intermediate loops completely
            for loop_idx in range(start_idx + 1, end_idx):
                loop_type = loop_order[loop_idx]
                if loop_type == LoopType.SETUP:
                    loop_ops = state.algorithm.setup_loop
                elif loop_type == LoopType.FORWARD:
                    loop_ops = state.algorithm.forward_loop
                else:
                    loop_ops = state.algorithm.update_loop
                    
                for operation in loop_ops:
                    if operation.operand1 == var or operation.operand2 == var:
                        return True
                        
            # Check operations in end loop before end_pos
            if end_loop == LoopType.SETUP:
                end_loop_ops = state.algorithm.setup_loop
            elif end_loop == LoopType.FORWARD:
                end_loop_ops = state.algorithm.forward_loop
            else:
                end_loop_ops = state.algorithm.update_loop
                
            for i in range(0, end_pos + 1):
                operation = end_loop_ops[i]
                if operation.operand1 == var or operation.operand2 == var:
                    return True
                    
        return False

    def _is_variable_overwritten_after_position(self, state: AlgorithmState, var: OperandType,
                                               loop_type: LoopType, position: int) -> bool:
        """
        Check if a variable is overwritten (used as target) after the given position.
        """
        # Get the appropriate loop
        if loop_type == LoopType.SETUP:
            loop_ops = state.algorithm.setup_loop
        elif loop_type == LoopType.FORWARD:
            loop_ops = state.algorithm.forward_loop
        else:
            loop_ops = state.algorithm.update_loop
            
        # Check operations after this position in the same loop
        for i in range(position + 1, len(loop_ops)):
            operation = loop_ops[i]
            if operation.target == var:
                return True
                
        # Check if overwritten in later loops (following execution order)
        if loop_type == LoopType.SETUP:
            # Check forward and update loops
            for operation in state.algorithm.forward_loop:
                if operation.target == var:
                    return True
            for operation in state.algorithm.update_loop:
                if operation.target == var:
                    return True
        elif loop_type == LoopType.FORWARD:
            # Check update loop
            for operation in state.algorithm.update_loop:
                if operation.target == var:
                    return True
                    
        return False

    def _remove_operation_at_position(self, state: AlgorithmState, loop_type: LoopType, position: int):
        """
        Remove operation at the specified position and update variable states.
        """
        state.algorithm.remove_operation(loop_type, position)
        
        # Shift variable positions that were after the deleted operation
        state._shift_variable_positions(loop_type, position+1, -1)

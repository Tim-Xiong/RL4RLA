"""Grammar for action space."""

from typing import Dict, List, Tuple

from domain.operation import Operation, OperatorType, OperandType
from domain.action import Action, LoopType, ActionType
from solver.variables import AlgorithmState, VariableType, AlgorithmStateManager


class OperationSpec:
    """Enhanced specification for operation type including type rules and loop constraints."""
    def __init__(self,
                 operand1_types: List[OperandType],
                 operand2_types: List[OperandType],
                 target_types: List[OperandType],
                 input_var_types: Tuple[VariableType, VariableType],
                 output_var_type: VariableType,
                 exclude_same_operands: bool = False,
                 is_commutative: bool = False,
                 allowed_loops: List[LoopType] = None):
        self.operand1_types = operand1_types
        self.operand2_types = operand2_types
        self.target_types = target_types
        self.input_var_types = input_var_types  # (left_input_type, right_input_type)
        self.output_var_type = output_var_type
        self.exclude_same_operands = exclude_same_operands
        self.is_commutative = is_commutative
        self.allowed_loops = allowed_loops if allowed_loops is not None else [LoopType.SETUP, LoopType.FORWARD, LoopType.UPDATE]


class ActionGenerator:
    """Generates legal actions with complete constraint validation.
    
    Responsibilities:
    - Define operation specifications and constraints
    - Generate valid actions based on state and constraints
    - Validate type compatibility and operand availability
    - Handle commutative operations and exclusion rules
    """
    
    def __init__(self, initial_symmetric_operands=None):
        self.operation_specs = self._create_operation_specs()
        self.state_manager = AlgorithmStateManager()
        self.initial_symmetric_operands = initial_symmetric_operands or set()

    def _update_symmetry_from_operation(self, symmetric, operation):
        """Update symmetry set based on a single operation."""
        op, op1, op2, target = operation.operator, operation.operand1, operation.operand2, operation.target

        # Only track matrix operands
        if target not in [OperandType.R1, OperandType.R2, OperandType.R3, OperandType.A]:
            return

        # A^T @ A or A @ A^T is symmetric
        if (op == OperatorType.MAT_TRANS_MAT_MUL or op == OperatorType.MAT_MAT_TRANS_MUL) and op1 == op2:
            symmetric.add(target)
        # Inverse of symmetric is symmetric
        elif op == OperatorType.MAT_INV and op1 in symmetric:
            symmetric.add(target)
        # Everything else: target is not symmetric
        else:
            symmetric.discard(target)

    def _get_symmetric_operands_at_position(self, state, loop_type, position):
        """Compute which operands are symmetric at a specific insertion position."""
        symmetric = self.initial_symmetric_operands.copy()

        # Apply operations in execution order before the insertion point
        if loop_type == LoopType.SETUP:
            for op in state.algorithm.setup_loop[:position]:
                self._update_symmetry_from_operation(symmetric, op)
        elif loop_type == LoopType.FORWARD:
            for op in state.algorithm.setup_loop:
                self._update_symmetry_from_operation(symmetric, op)
            for op in state.algorithm.forward_loop[:position]:
                self._update_symmetry_from_operation(symmetric, op)
        else:  # UPDATE
            for op in state.algorithm.setup_loop:
                self._update_symmetry_from_operation(symmetric, op)
            for op in state.algorithm.forward_loop:
                self._update_symmetry_from_operation(symmetric, op)
            for op in state.algorithm.update_loop[:position]:
                self._update_symmetry_from_operation(symmetric, op)

        return symmetric

    def _create_operation_specs(self) -> Dict[OperatorType, OperationSpec]:
        """Define operation specifications with both operand constraints and type rules."""
        return {
            # Vector operations
            OperatorType.VEC_VEC_ADD: OperationSpec(
                [OperandType.V1, OperandType.B, OperandType.X_T],
                [OperandType.V1, OperandType.B, OperandType.X_T],
                [OperandType.V1],
                (VariableType.VECTOR, VariableType.VECTOR),
                VariableType.VECTOR,
                is_commutative=True,
                exclude_same_operands=True
            ),
            OperatorType.VEC_VEC_SUB: OperationSpec(
                [OperandType.V1, OperandType.B, OperandType.X_T],
                [OperandType.V1, OperandType.B, OperandType.X_T],
                [OperandType.V1],
                (VariableType.VECTOR, VariableType.VECTOR),
                VariableType.VECTOR,
                exclude_same_operands=True
            ),
            # OperatorType.VEC_VEC_DOT: OperationSpec(
            #     [OperandType.V1, OperandType.B, OperandType.X_T],
            #     [OperandType.V1, OperandType.B, OperandType.X_T],
            #     [OperandType.A1],
            #     (VariableType.VECTOR, VariableType.VECTOR),
            #     VariableType.SCALAR,
            #     is_commutative=True
            # ),

            # Matrix-vector operations (NOT commutative)
            OperatorType.MAT_VEC_MUL: OperationSpec(
                [OperandType.A, OperandType.R1],
                [OperandType.V1, OperandType.B, OperandType.X_T],
                [OperandType.V1],
                (VariableType.MATRIX, VariableType.VECTOR),
                VariableType.VECTOR
            ),
            OperatorType.VEC_MAT_MUL: OperationSpec(
                [OperandType.V1, OperandType.B, OperandType.X_T],
                [OperandType.A, OperandType.R1],
                [OperandType.V1],
                (VariableType.VECTOR, VariableType.MATRIX),
                VariableType.VECTOR
            ),

            # Scalar operations
            # OperatorType.SCALAR_VEC_MUL: OperationSpec(
            #     [OperandType.A1],
            #     [OperandType.V1, OperandType.B, OperandType.X_T],
            #     [OperandType.V1],
            #     (VariableType.SCALAR, VariableType.VECTOR),
            #     VariableType.VECTOR,
            #     is_commutative=True
            # ),
            # OperatorType.SCALAR_DIV: OperationSpec(
            #     [OperandType.A1],
            #     [OperandType.A1],
            #     [OperandType.A1],
            #     (VariableType.SCALAR, VariableType.SCALAR),
            #     VariableType.SCALAR
            # ),

            # Matrix operations
            OperatorType.SKETCH: OperationSpec(
                [OperandType.A, OperandType.R1],
                [OperandType.NONE],
                [OperandType.R1],
                (VariableType.MATRIX, VariableType.NONE),
                VariableType.MATRIX,
                allowed_loops=[LoopType.SETUP]
            ),
            OperatorType.MAT_MAT_MUL: OperationSpec(
                [OperandType.A, OperandType.R1],
                [OperandType.A, OperandType.R1],
                [OperandType.R1],
                (VariableType.MATRIX, VariableType.MATRIX),
                VariableType.MATRIX
            ),
            OperatorType.HHQR: OperationSpec(
                [OperandType.A, OperandType.R1],
                [OperandType.NONE],
                [OperandType.R1],
                (VariableType.MATRIX, VariableType.NONE),
                VariableType.MATRIX
            ),
            # OperatorType.TRIANGULAR_SOLVE: OperationSpec(
            #     [OperandType.A, OperandType.R1],
            #     [OperandType.V1, OperandType.B, OperandType.X_T],
            #     [OperandType.V1],
            #     (VariableType.MATRIX, VariableType.VECTOR),
            #     VariableType.VECTOR
            # ),
            OperatorType.MAT_INV: OperationSpec(
                [OperandType.R1],
                [OperandType.NONE],
                [OperandType.R1],
                (VariableType.MATRIX, VariableType.NONE),
                VariableType.MATRIX
            ),
            OperatorType.MAT_MAT_TRANS_MUL: OperationSpec(
                [OperandType.A, OperandType.R1],
                [OperandType.A, OperandType.R1],
                [OperandType.R1],
                (VariableType.MATRIX, VariableType.MATRIX),
                VariableType.MATRIX
            ),
            # OperatorType.MAT_TRANS_MAT_MUL: OperationSpec(
            #     [OperandType.A, OperandType.R1],
            #     [OperandType.A, OperandType.R1],
            #     [OperandType.R1],
            #     (VariableType.MATRIX, VariableType.MATRIX),
            #     VariableType.MATRIX
            # ),
            
            # Special operations
            OperatorType.LEVERAGE_SCORE: OperationSpec(
                [OperandType.A, OperandType.R1],
                [OperandType.NONE],
                [OperandType.V3],
                (VariableType.MATRIX, VariableType.NONE),
                VariableType.VECTOR,
                allowed_loops=[LoopType.SETUP]
            ),
            OperatorType.SUBSAMPLING: OperationSpec(
                [OperandType.V3],
                [OperandType.NONE],
                [OperandType.NONE],
                (VariableType.VECTOR, VariableType.NONE),
                VariableType.NONE,
                allowed_loops=[LoopType.SETUP, LoopType.FORWARD]
            ),
            # Control operations
            # OperatorType.DELETE: OperationSpec(
            #     [OperandType.NONE],
            #     [OperandType.NONE],
            #     [OperandType.NONE],
            #     (VariableType.NONE, VariableType.NONE),
            #     VariableType.NONE
            # ),
            # OperatorType.DO_NOTHING: OperationSpec(
            #     [OperandType.NONE],
            #     [OperandType.NONE],
            #     [OperandType.NONE],
            #     (VariableType.NONE, VariableType.NONE),
            #     VariableType.NONE
            # ),
        }

    def get_legal_actions(self, state: AlgorithmState, setup_loop: bool, forward_loop: bool, update_loop: bool) -> List[Action]:
        """Generate all legal actions for the current state using INSERT actions only."""
        legal_actions = []

        loops = []
        if setup_loop:
            loops.append(LoopType.SETUP)
        if forward_loop:
            loops.append(LoopType.FORWARD)
        if update_loop:
            loops.append(LoopType.UPDATE)

        # Generate INSERT actions for all enabled loops
        for loop_type in loops:
            # Determine possible insertion positions
            if loop_type == LoopType.SETUP:
                max_position = len(state.algorithm.setup_loop)
            elif loop_type == LoopType.FORWARD:
                max_position = len(state.algorithm.forward_loop)
            else:
                max_position = len(state.algorithm.update_loop)

            # Generate INSERT actions for all valid positions (0 to length inclusive)
            # Position 0 = insert at beginning, position = length = append at end
            for position in range(max_position + 1):
                legal_actions.extend(self._generate_legal_actions_for_loop(
                    state, loop_type, ActionType.INSERT, position
                ))

        # Filter LEVERAGE_SCORE: only one allowed, at lowest position before SUBSAMPLING
        lev_actions = [a for a in legal_actions if a.operation.operator == OperatorType.LEVERAGE_SCORE]
        if lev_actions:
            # Check if LEVERAGE_SCORE already exists
            if any(op.operator == OperatorType.LEVERAGE_SCORE for op in state.algorithm.setup_loop):
                lev_actions = []  # Remove all if already exists
            else:
                # Find SUBSAMPLING position
                sub_pos = next((i for i, op in enumerate(state.algorithm.setup_loop)
                               if op.operator == OperatorType.SUBSAMPLING), None)
                if sub_pos is None and any(op.operator == OperatorType.SUBSAMPLING
                                           for op in state.algorithm.forward_loop):
                    sub_pos = len(state.algorithm.setup_loop)

                # Keep only before SUBSAMPLING and at lowest position
                if sub_pos is not None:
                    lev_actions = [a for a in lev_actions if a.position < sub_pos]
                if lev_actions:
                    min_pos = min(a.position for a in lev_actions)
                    lev_actions = [a for a in lev_actions if a.position == min_pos]

            # Replace LEVERAGE_SCORE actions
            legal_actions = [a for a in legal_actions if a.operation.operator != OperatorType.LEVERAGE_SCORE]
            legal_actions.extend(lev_actions)

        return legal_actions

    def _generate_legal_actions_for_loop(self, state: AlgorithmState, loop_type: LoopType,
                                       action_type: ActionType, position: int) -> List[Action]:
        """Generate legal actions for a specific loop and action type using direct validation."""
        legal_actions = []

        # Compute which operands are symmetric at this insertion point
        symmetric_at_position = self._get_symmetric_operands_at_position(state, loop_type, position)

        for operator, spec in self.operation_specs.items():
            # Check if operator is allowed in this loop
            if loop_type not in spec.allowed_loops:
                continue
                
            # Directly generate valid combinations using spec constraints and state validation
            for operand1 in spec.operand1_types:
                if not self._is_operand_available(state, operand1, spec.input_var_types[0], loop_type, position):
                    continue

                for operand2 in spec.operand2_types:
                    if not self._is_operand_available(state, operand2, spec.input_var_types[1], loop_type, position):
                        continue

                    # Skip same operands if specified
                    if spec.exclude_same_operands and operand1 == operand2:
                        continue

                    # For commutative operations, only generate one equivalent action
                    # by ensuring operand1 <= operand2 (lexicographically by enum value)
                    if spec.is_commutative and operand1.value > operand2.value:
                        continue

                    # Skip redundant operations for symmetric matrices
                    if operator == OperatorType.VEC_MAT_MUL and operand2 in symmetric_at_position:
                        continue
                    if operator == OperatorType.MAT_MAT_TRANS_MUL and operand2 in symmetric_at_position:
                        continue

                    for target in spec.target_types:
                        if not self._is_target_available(state, target, spec.output_var_type, loop_type, position):
                            continue
                            
                        action = Action(
                            operation=Operation(
                                operator=operator,
                                operand1=operand1,
                                operand2=operand2,
                                target=target
                            ),
                            loop_select=loop_type,
                            action_type=action_type,
                            position=position
                        )
                        legal_actions.append(action)
        
        return legal_actions

    def _is_operand_available(self, state: AlgorithmState, operand: OperandType,
                             expected_type: VariableType, loop_type: LoopType, position: int) -> bool:
        """Check if an operand is available and has the correct type."""
        # Check if operand can be used at this position
        if not state.can_use_as_operand(operand, loop_type, position):
            return False
        
        # Check type compatibility
        var_type = state.get_variable_type(operand)
        return var_type == expected_type
    
    def _is_target_available(self, state: AlgorithmState, target: OperandType,
                            expected_type: VariableType, loop_type: LoopType, position: int) -> bool:
        """Check if a target is available and has the correct type."""
        # Check if target can be used at this position
        if not state.can_use_as_target(target, loop_type, position):
            return False
        
        # Check type compatibility
        var_type = state.get_variable_type(target)
        return var_type == expected_type

    def is_action_valid(self, action: Action, state: AlgorithmState) -> bool:
        """Check if an action is valid for the given algorithm state."""
        # Get the operation spec for this action (O(1) lookup)
        operator = action.operation.operator
        if operator not in self.operation_specs:
            return False
        
        spec = self.operation_specs[operator]
        
        # Check if operator is allowed in this loop
        if action.loop_select not in spec.allowed_loops:
            return False
        
        # Validate operands
        if not self._is_operand_available(state, action.operation.operand1,
                                         spec.input_var_types[0], action.loop_select, action.position):
            return False
        
        if not self._is_operand_available(state, action.operation.operand2,
                                         spec.input_var_types[1], action.loop_select, action.position):
            return False
        
        # Validate target
        if not self._is_target_available(state, action.operation.target,
                                        spec.output_var_type, action.loop_select, action.position):
            return False
        
        # Check spec-specific constraints
        if spec.exclude_same_operands and action.operation.operand1 == action.operation.operand2:
            return False
        
        # Check if operands are in allowed lists
        if (action.operation.operand1 not in spec.operand1_types or
            action.operation.operand2 not in spec.operand2_types or
            action.operation.target not in spec.target_types):
            return False
        
        return True

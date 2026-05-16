"""Algorithm execution engine."""

from typing import List, Dict, Any, Optional
import numpy as np
import scipy.linalg as la

from domain.linear_system import LinearSystem
from domain.algorithm import Algorithm
from domain.operation import Operation, OperatorType, OperandType
from domain.metrics import ExecutionResult
from infrastructure.linear_algebra.operations import LinearAlgebraOperations
from infrastructure.linear_algebra.data_generation import SamplingOperations


class AlgorithmExecutor:
    """Executes algorithms on linear systems."""
    
    def __init__(self,
                 linear_system: LinearSystem,
                 linear_algebra_ops: LinearAlgebraOperations,
                 max_iterations: int = 30,
                 learning_rate: float = 0.2,
                 batch_size: int = 100,
                 cost_cap: float = 1e9,
                 random_seed: Optional[int] = None):

        self.linear_system = linear_system
        self.linear_algebra_ops = linear_algebra_ops
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cost_cap = cost_cap
        self.rng = np.random.default_rng(random_seed)

        # Initialize operand storage
        self.reset_operands()
    
    def reset_operands(self):
        """Reset operand storage to initial state."""
        # Initialize cache with random values for unused operands
        rng = self.rng
        dim = self.linear_system.num_cols
        
        self.operands = {
            # Vector operands
            OperandType.V1: rng.random(dim) * 1e10,
            # OperandType.V2: rng.random(dim) * 1e10,
            OperandType.V3: np.ones(self.linear_system.num_rows) / self.linear_system.num_rows,
            
            # Scalar operands
            # OperandType.A1: rng.random() * 1e10,
            # OperandType.A2: rng.random() * 1e10,
            
            # System operands
            OperandType.B: self.linear_system.vector_b.copy(),
            OperandType.A: self.linear_system.matrix_A.copy(),
            OperandType.LR: self.learning_rate,
            OperandType.X_T: np.zeros(dim),
            OperandType.UPDATE: np.zeros(dim),
            
            # Matrix operands
            OperandType.R1: rng.random((dim, dim)) * 1e10,
            # OperandType.R2: np.eye(dim),
            # OperandType.R3: np.eye(dim),
        }
    
    def execute_algorithm(self, algorithm: Algorithm, lr_sweep: bool = False) -> ExecutionResult:
        """Execute an algorithm and return results."""
        if lr_sweep:
            lr_list = [0.1, 0.2, 0.5, 0.7, 1.0]
            original_lr = self.learning_rate
            best_result = ExecutionResult(success=False)
            best_loss = float('inf')
            
            for lr in lr_list:
                self.learning_rate = lr
                result = self.execute_algorithm(algorithm, lr_sweep=False)
                if result.success and result.loss < best_loss:
                    best_loss = result.loss
                    best_result = result
            
            self.learning_rate = original_lr
            return best_result
        
        try:
            self.reset_operands()
            
            # Track execution metrics
            total_cost = 0.0
            convergence_history = []
            final_R1 = None
            
            # Execute setup loop operations
            for operation in algorithm.setup_loop:
                success, cost = self._execute_operation(operation)
                if not success:
                    return ExecutionResult(success=False)
                total_cost += cost
            
            # Execute iterative forward + update loops
            for iteration in range(self.max_iterations):

                ### DEBUG ###
                # res = np.linalg.norm(self.linear_system.matrix_A @ self.operands[OperandType.X_T] - self.linear_system.vector_b)

                iteration_cost = 0.0
                
                # Execute forward loop operations
                for operation in algorithm.forward_loop:
                    success, cost = self._execute_operation(operation)
                    if not success:
                        return ExecutionResult(success=False)
                    iteration_cost += cost
                
                # Execute update loop operations
                for operation in algorithm.update_loop:
                    success, cost = self._execute_operation(operation)
                    if not success:
                        return ExecutionResult(success=False)
                    iteration_cost += cost
                
                # Update step: update = lr * v1
                success, cost = self._execute_update_step()
                if not success:
                    return ExecutionResult(success=False)
                iteration_cost += cost
                
                # Apply update: x_t = x_t - update
                success, cost = self._execute_apply_update()
                if not success:
                    return ExecutionResult(success=False)
                iteration_cost += cost
                
                total_cost += iteration_cost
                
                # Record convergence
                current_x = self.operands[OperandType.X_T].copy()
                convergence_history.append(current_x)

            # Calculate final metrics
            final_solution = self.operands[OperandType.X_T]
            final_loss = self._calculate_loss(final_solution)
            max_residual_ratio = self._calculate_max_residual_ratio(convergence_history)
            
            # Get condition number if R1 was computed
            condition_number = None
            if OperandType.R1 in self.operands:
                final_R1 = self.operands[OperandType.R1]
                try:
                    ATA = self.linear_system.matrix_A.T @ self.linear_system.matrix_A
                    if ATA.shape == final_R1.shape:
                        condition_number = float(np.linalg.cond(ATA @ final_R1))
                except Exception:
                    pass
            
            return ExecutionResult(
                success=True,
                final_solution=final_solution,
                loss=final_loss,
                convergence_history=convergence_history,
                computational_cost=total_cost,
                condition_number=condition_number,
                max_residual_ratio=max_residual_ratio
            )
            
        except Exception as e:
            print(f"Algorithm execution failed: {e}")
            return ExecutionResult(success=False)
    
    def _execute_operation(self, operation: Operation) -> tuple:
        """Execute a single operation. Returns (success, cost)."""
        # Get operand data
        operand1_data = self._get_operand_data(operation.operand1)
        operand2_data = self._get_operand_data(operation.operand2)

        # Special handling for subsampling operation
        if operation.operator == OperatorType.SUBSAMPLING:
            return self._execute_subsampling(operation.operand1)

        # Execute operation using linear algebra operations
        success, cost, result = self.linear_algebra_ops.execute_operation(
            operation.operator,
            operand1_data,
            operand2_data,
            operation.target,
            cost_cap=self.cost_cap
        )

        if success and result is not None:
            # Store result in target operand
            self.operands[operation.target] = result

        return success, cost
    
    def _execute_subsampling(self, probability_vector_operand: OperandType) -> tuple:
        """Execute subsampling operation."""
        try:
            prob_vector = self.operands[probability_vector_operand]
            
            # Check if probability vector is valid
            if np.any(prob_vector < 0):
                return False, 0.0
            
            # Create sampling matrix
            batch_size = min(self.batch_size, self.linear_system.num_rows)
            indices = SamplingOperations.sample_rows(
                self.linear_system.num_rows,
                batch_size,
                prob_vector,
                self.rng
            )
            
            S_t = SamplingOperations.create_sampling_matrix(
                indices,
                self.linear_system.num_rows,
                batch_size
            )

            # # with reweighting
            # S_t = SamplingOperations.create_sampling_matrix(
            #     self.linear_system.num_rows,
            #     batch_size,
            #     prob_vector,
            #     self.rng
            # )
            
            # Apply sampling to A and b
            self.operands[OperandType.A] = S_t @ self.linear_system.matrix_A
            self.operands[OperandType.B] = S_t @ self.linear_system.vector_b
            
            # Calculate cost
            cost = batch_size * self.linear_system.num_cols
            
            return True, cost
            
        except Exception as e:
            return False, 0.0
    
    def _execute_update_step(self) -> tuple:
        """Execute update = lr * v1."""
        try:
            lr = self.operands[OperandType.LR]
            v1 = self.operands[OperandType.V1]
            
            update = lr * v1
            self.operands[OperandType.UPDATE] = update
            
            cost = v1.shape[0]  # Cost of scalar-vector multiplication
            return True, cost
            
        except Exception:
            return False, 0.0
    
    def _execute_apply_update(self) -> tuple:
        """Execute x_t = x_t - update."""
        try:
            x_t = self.operands[OperandType.X_T]
            update = self.operands[OperandType.UPDATE]
            
            new_x_t = x_t - update
            self.operands[OperandType.X_T] = new_x_t
            
            cost = x_t.shape[0]  # Cost of vector subtraction
            return True, cost
            
        except Exception:
            return False, 0.0
    
    def _get_operand_data(self, operand: OperandType):
        """Get data for an operand."""
        if operand == OperandType.NONE:
            return None
        return self.operands.get(operand)
    
    def _calculate_loss(self, solution: np.ndarray) -> float:
        """Calculate normalized residual loss."""
        try:
            residual = self.linear_system.matrix_A @ solution - self.linear_system.vector_b
            residual_norm = la.norm(residual)
            b_norm = la.norm(self.linear_system.vector_b)
            
            if b_norm == 0:
                return float(residual_norm)
            
            return float(residual_norm / b_norm)
            
        except Exception:
            return float('inf')
    
    def _calculate_max_residual_ratio(self, convergence_history: List[np.ndarray]) -> float:
        """Calculate maximum ratio of consecutive residual norms."""
        if len(convergence_history) < 2:
            return 1.0
        
        try:
            # Calculate residual norms for each solution
            residual_norms = []
            for x in convergence_history:
                residual = self.linear_system.matrix_A @ x - self.linear_system.vector_b
                norm = la.norm(residual) / la.norm(self.linear_system.vector_b)
                residual_norms.append(norm)
            
            # Calculate ratios
            ratios = []
            for i in range(1, len(residual_norms)):
                if residual_norms[i-1] > 0:
                    ratio = residual_norms[i] / residual_norms[i-1]
                    ratios.append(ratio)
            
            return max(ratios) if ratios else 1.0
            
        except Exception:
            return 1.0
    
    def get_current_operand_values(self) -> Dict[str, Any]:
        """Get current values of all operands for debugging."""
        result = {}
        for operand_type in OperandType:
            if operand_type in self.operands:
                value = self.operands[operand_type]
                if isinstance(value, np.ndarray):
                    result[operand_type.name] = {
                        "shape": value.shape,
                        "norm": float(np.linalg.norm(value)) if value.size > 0 else 0.0,
                        "mean": float(np.mean(value)) if value.size > 0 else 0.0
                    }
                else:
                    result[operand_type.name] = float(value)
        return result
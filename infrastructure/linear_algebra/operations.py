"""Linear algebra operations implementation."""

import warnings
from typing import Tuple, Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as spar

from domain.operation import OperatorType, OperandType


class LinearAlgebraOperations:
    """Handles execution of mathematical operations on matrices and vectors."""

    def __init__(self, sketch_matrix: Optional[spar.csc_matrix] = None):
        self.sketch_matrix = sketch_matrix

    def execute_operation(self,
                         operator: OperatorType,
                         operand1_data,
                         operand2_data,
                         target_operand: OperandType,
                         cost_cap: Optional[float] = None) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Execute a mathematical operation.

        Args:
            operator: The operation to perform
            operand1_data: First operand data
            operand2_data: Second operand data
            target_operand: Target operand for result
            cost_cap: Maximum allowed computational cost. If operation cost exceeds this,
                     execution is skipped and (False, estimated_cost, None) is returned.

        Returns:
            Tuple of (success, computational_cost, result)
        """

        # Calculate cost once upfront
        cost = self._calculate_operation_cost(operator, operand1_data, operand2_data)

        # Check cost cap before executing expensive operations
        if cost_cap is not None and cost > cost_cap:
            return False, cost, None

        # Check for numerical overflow in operands
        # if self._has_numerical_overflow(operand1_data) or self._has_numerical_overflow(operand2_data):
        #     warnings.warn(f"Numerical overflow in operands")
        #     return False, 0.0, None

        try:
            # Execute the operation (without cost calculation)
            if operator == OperatorType.DO_NOTHING:
                success, result = True, None

            elif operator == OperatorType.VEC_VEC_ADD:
                success, result = self._vector_vector_add(operand1_data, operand2_data)

            elif operator == OperatorType.VEC_VEC_SUB:
                success, result = self._vector_vector_sub(operand1_data, operand2_data)

            elif operator == OperatorType.VEC_VEC_DOT:
                success, result = self._vector_vector_dot(operand1_data, operand2_data)

            elif operator == OperatorType.MAT_VEC_MUL:
                success, result = self._matrix_vector_mul(operand1_data, operand2_data)

            elif operator == OperatorType.VEC_MAT_MUL:
                success, result = self._vector_matrix_mul(operand1_data, operand2_data)

            elif operator == OperatorType.SCALAR_VEC_MUL:
                success, result = self._scalar_vector_mul(operand1_data, operand2_data)

            elif operator == OperatorType.SCALAR_DIV:
                success, result = self._scalar_div(operand1_data, operand2_data)

            elif operator == OperatorType.SKETCH:
                success, result = self._sketch_operation(operand1_data)

            elif operator == OperatorType.MAT_MAT_MUL:
                success, result = self._matrix_matrix_mul(operand1_data, operand2_data)

            elif operator == OperatorType.MAT_MAT_TRANS_MUL:
                success, result = self._matrix_matrix_trans_mul(operand1_data, operand2_data)

            elif operator == OperatorType.MAT_TRANS_MAT_MUL:
                success, result = self._matrix_trans_matrix_mul(operand1_data, operand2_data)

            elif operator == OperatorType.HHQR:
                success, result = self._householder_qr(operand1_data)

            elif operator == OperatorType.TRIANGULAR_SOLVE:
                success, result = self._triangular_solve(operand1_data, operand2_data)

            elif operator == OperatorType.MAT_INV:
                success, result = self._matrix_inverse(operand1_data)

            elif operator == OperatorType.LEVERAGE_SCORE:
                success, result = self._compute_leverage_scores(operand1_data)

            else:
                success, result = False, None

            # Return with precomputed cost
            return success, cost, result

        except Exception as e:
            warnings.warn(f"Error executing operation {operator}: \n{e}")
            return False, cost, None

    def _calculate_operation_cost(self, operator: OperatorType, operand1_data, operand2_data) -> float:
        """
        Calculate computational cost of an operation without executing it.

        Args:
            operator: The operation type
            operand1_data: First operand data
            operand2_data: Second operand data

        Returns:
            FLOP count for the operation
        """
        try:
            if operator == OperatorType.DO_NOTHING:
                return 0.0

            elif operator in (OperatorType.VEC_VEC_ADD, OperatorType.VEC_VEC_SUB):
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_vector_op_cost(operand1_data)

            elif operator == OperatorType.VEC_VEC_DOT:
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_dot_product_cost(operand1_data)

            elif operator == OperatorType.MAT_VEC_MUL:
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_matvec_cost(operand1_data, operand2_data)

            elif operator == OperatorType.VEC_MAT_MUL:
                if operand2_data is None or not hasattr(operand2_data, 'shape'):
                    return 0.0
                return self._calculate_vecmat_cost(operand1_data, operand2_data)

            elif operator == OperatorType.SCALAR_VEC_MUL:
                if operand2_data is None or not hasattr(operand2_data, 'shape'):
                    return 0.0
                return self._calculate_vector_op_cost(operand2_data)

            elif operator == OperatorType.SCALAR_DIV:
                return 1.0

            elif operator == OperatorType.SKETCH:
                if self.sketch_matrix is None or operand1_data is None:
                    return 0.0
                if not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_sketch_cost(operand1_data)

            elif operator == OperatorType.MAT_MAT_MUL:
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_matmat_cost(operand1_data, operand2_data)

            elif operator == OperatorType.MAT_MAT_TRANS_MUL:
                if operand1_data is None or operand2_data is None:
                    return 0.0
                if not hasattr(operand1_data, 'shape') or not hasattr(operand2_data, 'shape'):
                    return 0.0
                cost = self._calculate_matmat_cost(operand1_data, operand2_data)
                cost += operand2_data.shape[-1] * operand2_data.shape[-2]
                return cost

            elif operator == OperatorType.MAT_TRANS_MAT_MUL:
                if operand1_data is None or operand2_data is None:
                    return 0.0
                if not hasattr(operand1_data, 'shape') or not hasattr(operand2_data, 'shape'):
                    return 0.0
                cost = self._calculate_matmat_cost(operand1_data, operand2_data)
                cost += operand1_data.shape[-1] * operand1_data.shape[-2]
                return cost

            elif operator == OperatorType.HHQR:
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_hhqr_cost(operand1_data)

            elif operator == OperatorType.TRIANGULAR_SOLVE:
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_triangular_solve_cost(operand1_data)

            elif operator == OperatorType.MAT_INV:
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_inverse_cost(operand1_data)

            elif operator == OperatorType.LEVERAGE_SCORE:
                if operand1_data is None or not hasattr(operand1_data, 'shape'):
                    return 0.0
                return self._calculate_leverage_score_cost(operand1_data)

            else:
                return 0.0

        except Exception:
            # If cost estimation fails, return 0 to allow normal execution path
            return 0.0

    def _has_numerical_overflow(self, data) -> bool:
        """Check for numerical overflow in array data."""
        if data is None:
            return False
        if isinstance(data, np.ndarray):
            return np.max(np.abs(data)) >= 1e8
        return False

    def _vector_vector_add(self, v1, v2) -> Tuple[bool, np.ndarray]:
        """Vector addition: v1 + v2."""
        if v1.shape != v2.shape:
            return False, None
        result = v1 + v2
        return True, result

    def _vector_vector_sub(self, v1, v2) -> Tuple[bool, np.ndarray]:
        """Vector subtraction: v1 - v2."""
        if v1.shape != v2.shape:
            return False, None
        result = v1 - v2
        return True, result

    def _vector_vector_dot(self, v1, v2) -> Tuple[bool, float]:
        """Vector dot product: v1 · v2."""
        if v1.shape != v2.shape:
            return False, None
        result = np.dot(v1, v2)
        return True, result

    def _matrix_vector_mul(self, matrix, vector) -> Tuple[bool, np.ndarray]:
        """Matrix-vector multiplication: A @ v."""
        if matrix.shape[-1] != vector.shape[-1]:
            return False, None
        result = matrix @ vector
        return True, result

    def _vector_matrix_mul(self, vector, matrix) -> Tuple[bool, np.ndarray]:
        """Vector-matrix multiplication: A.T @ v."""
        if vector.shape[-1] != matrix.shape[-2]:
            return False, None
        result = vector @ matrix
        return True, result

    def _scalar_vector_mul(self, scalar, vector) -> Tuple[bool, np.ndarray]:
        """Scalar-vector multiplication: s * v."""
        result = scalar * vector
        return True, result

    def _scalar_div(self, scalar1, scalar2) -> Tuple[bool, float]:
        """Scalar division: s1 / s2."""
        epsilon = 1e-10
        result = scalar1 / (scalar2 + epsilon)
        return True, result

    def _sketch_operation(self, matrix) -> Tuple[bool, np.ndarray]:
        """Apply sketching matrix: S @ A."""
        if self.sketch_matrix is None:
            return False, None
        if self.sketch_matrix.shape[-1] != matrix.shape[-2]:
            return False, None

        result = self.sketch_matrix @ matrix
        return True, result

    def _matrix_matrix_mul(self, matrix1, matrix2) -> Tuple[bool, np.ndarray]:
        """Matrix-matrix multiplication: A @ B."""
        if matrix1.shape[-1] != matrix2.shape[-2]:
            return False, None
        result = matrix1 @ matrix2
        return True, result

    def _matrix_matrix_trans_mul(self, matrix1, matrix2) -> Tuple[bool, np.ndarray]:
        """Matrix-matrix^T multiplication: A @ B^T."""
        if matrix1.shape[-1] != matrix2.shape[-1]:
            return False, None
        result = matrix1 @ matrix2.T
        return True, result

    def _matrix_trans_matrix_mul(self, matrix1, matrix2) -> Tuple[bool, np.ndarray]:
        """Matrix^T-matrix multiplication: A^T @ B."""
        if matrix1.shape[-2] != matrix2.shape[-2]:
            return False, None
        result = matrix1.T @ matrix2
        return True, result

    def _householder_qr(self, matrix) -> Tuple[bool, np.ndarray]:
        """Householder QR decomposition returning R matrix."""
        if matrix.shape[-2] < matrix.shape[-1]:
            return False, None

        _, R = la.qr(matrix, overwrite_a=True, mode="economic")
        return True, R

    def _triangular_solve(self, matrix, vector) -> Tuple[bool, np.ndarray]:
        """Solve triangular system: R @ x = b."""
        if not self._is_upper_triangular(matrix):
            return False, None
        if matrix.shape[-1] != matrix.shape[-2]:
            return False, None
        if matrix.shape[-1] != vector.shape[-1]:
            return False, None

        result = la.solve_triangular(matrix, vector, lower=False)
        return True, result

    def _matrix_inverse(self, matrix) -> Tuple[bool, np.ndarray]:
        """Matrix inversion: A^(-1)."""
        if matrix.shape[-1] != matrix.shape[-2]:
            return False, None

        if self._is_upper_triangular(matrix):
            # Efficient triangular solve
            result = la.solve_triangular(
                matrix,
                np.eye(matrix.shape[-1]),
                lower=False,
                check_finite=False,
                overwrite_b=True
            )
        else:
            # General matrix inverse
            result = la.pinv(matrix)

        return True, result

    def _compute_leverage_scores(self, matrix) -> Tuple[bool, np.ndarray]:
        """Compute leverage scores of matrix rows."""
        leverage_scores = np.sum(matrix ** 2, axis=1)
        result = leverage_scores / np.sum(leverage_scores)
        return True, result

    def _is_upper_triangular(self, matrix) -> bool:
        """Check if matrix is upper triangular."""
        if matrix.shape[0] != matrix.shape[1]:
            return False
        if spar.issparse(matrix):
            dense_matrix = matrix.toarray()
            return np.allclose(dense_matrix, np.triu(dense_matrix), atol=1e-10)
        elif isinstance(matrix, np.ndarray) and matrix.ndim == 2:
            return np.allclose(matrix, np.triu(matrix), atol=1e-10)
        return False

    def _calculate_vector_op_cost(self, v1) -> float:
        """Calculate cost for vector addition/subtraction/scalar multiply."""
        return float(v1.shape[-1])

    def _calculate_dot_product_cost(self, v1) -> float:
        """Calculate cost for dot product."""
        return 2.0 * v1.shape[-1]

    def _calculate_sketch_cost(self, matrix) -> float:
        """Calculate cost for sketch operation."""
        d = self.sketch_matrix.shape[0]
        n = matrix.shape[-1]
        return float(d * 8 * n)

    def _calculate_hhqr_cost(self, matrix) -> float:
        """Calculate cost for Householder QR."""
        return 2.0 * matrix.shape[-2] * matrix.shape[-1] * matrix.shape[-1]

    def _calculate_triangular_solve_cost(self, matrix) -> float:
        """Calculate cost for triangular solve."""
        return float(matrix.shape[-1] * matrix.shape[-1])

    def _calculate_inverse_cost(self, matrix) -> float:
        """Calculate cost for matrix inverse."""
        n = matrix.shape[-1]
        if self._is_upper_triangular(matrix):
            return float(n * n)
        return (2.0 / 3.0) * (n ** 3)

    def _calculate_leverage_score_cost(self, matrix) -> float:
        """Calculate cost for leverage score computation."""
        return 2.0 * matrix.shape[0] * matrix.shape[1]

    def _calculate_matvec_cost(self, matrix, vector) -> float:
        """Calculate cost of matrix-vector multiplication."""
        # Check if it's the sketch matrix (sparse)
        if hasattr(self, 'sketch_matrix') and matrix is self.sketch_matrix:
            return matrix.shape[0] * 8  # 8 is vec_nnz parameter
        return 2 * matrix.shape[-1] * matrix.shape[-2]

    def _calculate_vecmat_cost(self, vector, matrix) -> float:
        """Calculate cost of vector-matrix multiplication."""
        if hasattr(self, 'sketch_matrix') and matrix is self.sketch_matrix:
            return matrix.shape[1] * 8  # 8 is vec_nnz parameter
        return 2 * matrix.shape[-1] * matrix.shape[-2]

    def _calculate_matmat_cost(self, matrix1, matrix2) -> float:
        """Calculate cost of matrix-matrix multiplication."""
        # Handle sparse sketch matrix cases
        if hasattr(self, 'sketch_matrix'):
            if matrix1 is self.sketch_matrix:
                return matrix1.shape[0] * 8 * matrix2.shape[1]
            elif matrix2 is self.sketch_matrix:
                return matrix1.shape[0] * matrix1.shape[1] * 8

        # Check for triangular optimization
        if self._is_upper_triangular(matrix2):
            return (
                matrix1.shape[-2] * matrix1.shape[-1] * matrix2.shape[-1]
                - matrix1.shape[-2] * max(0, matrix1.shape[-1] - matrix2.shape[-1]) ** 2
                - matrix1.shape[-2] * max(matrix1.shape[-1], matrix2.shape[-1])
            )

        return 2 * matrix1.shape[-1] * matrix1.shape[-2] * matrix2.shape[-1]

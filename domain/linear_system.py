"""
Linear system domain model.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np


class SystemType(Enum):
    """Type of linear system."""
    PSD = "PSD"
    LOW_COND = "low_cond"
    MID_COND = "mid_cond"
    HIGH_COND = "high_cond"


@dataclass
class LinearSystem:
    """Represent a linear system Ax = b."""

    matrix_A: np.ndarray
    vector_b: np.ndarray
    solution_x: np.ndarray
    system_type: SystemType
    condition_number: Optional[float] = None

    def __post_init__(self):
        """Calculate condition number if not provided."""
        if self.condition_number is None:
            self.condition_number = np.linalg.cond(self.matrix_A)

    @property
    def num_rows(self) -> int:
        """Number of rows in matrix A."""
        return self.matrix_A.shape[0]

    @property
    def num_cols(self) -> int:
        """Number of columns in matrix A."""
        return self.matrix_A.shape[1]

    @property
    def residual_norm(self, x_t: np.ndarray) -> float:
        """Calculate residual norm ||Ax - b||."""
        return np.linalg.norm(self.matrix_A @ x_t - self.vector_b)

    @property
    def relative_error(self, x_t: np.ndarray) -> float:
        """Calculate relative error ||x_t - x_true|| / ||x_true||."""
        if np.linalg.norm(self.solution_x) == 0:
            return np.linalg.norm(x_t)
        return np.linalg.norm(x_t - self.solution_x) / np.linalg.norm(self.solution_x)

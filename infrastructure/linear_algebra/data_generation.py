"""Linear system data generation utilities."""

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as spar

from domain.linear_system import LinearSystem, SystemType


class LinearSystemFactory:
    """Factory for creating different types of linear systems."""

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.default_rng(random_seed)

    def create_system(self,
                     system_type: SystemType,
                     num_rows: int,
                     num_cols: int,
                     condition_number: float = 2.0,
                     prop_range: float = 1.0,
                     **kwargs) -> LinearSystem:
        """Create a linear system of the specified type."""

        if system_type == SystemType.PSD:
            return self._create_psd_system(num_rows, condition_number, **kwargs)
        elif system_type == SystemType.LOW_COND:
            return self._create_low_condition_system(num_rows, num_cols, prop_range, **kwargs)
        elif system_type == SystemType.MID_COND:
            return self._create_mid_condition_system(num_rows, num_cols, prop_range, **kwargs)
        elif system_type == SystemType.HIGH_COND:
            return self._create_high_condition_system(num_rows, num_cols, prop_range, **kwargs)
        else:
            raise ValueError(f"Unknown system type: {system_type}")

    def _create_psd_system(self,
                          size: int,
                          condition_number: float,
                          alpha: float = 1e-3,
                          fix_min: bool = False) -> LinearSystem:
        """Create a positive semi-definite system."""

        A = self._generate_pd_matrix(size, condition_number, alpha, fix_min)
        x = self.rng.standard_normal(size)
        b = A @ x

        return LinearSystem(
            matrix_A=A,
            vector_b=b,
            solution_x=x,
            system_type=SystemType.PSD
        )

    def _create_low_condition_system(self,
                                   num_rows: int,
                                   num_cols: int,
                                   prop_range: float,
                                   lev: str = "low",
                                   vt_dis: str = "gauss") -> LinearSystem:
        """Create a low condition number system."""

        # Create well-conditioned spectrum, cond=2=max_eigenvalue/min_eigenvalue
        min_eigenvalue = 1
        max_eigenvalue = 2
        scaling = np.linspace(0, 1, num_cols)
        spectrum = min_eigenvalue + scaling * (max_eigenvalue - min_eigenvalue)
        spectrum += 1e-6

        A, b, x_opt = self._create_system_from_spectrum(
            num_rows, num_cols, spectrum, prop_range, lev, vt_dis
        )

        return LinearSystem(
            matrix_A=A,
            vector_b=b,
            solution_x=x_opt,
            system_type=SystemType.LOW_COND
        )

    def _create_mid_condition_system(self,
                                    num_rows: int,
                                    num_cols: int,
                                    prop_range: float,
                                    lev: str = "high",
                                    vt_dis: str = "t_dist") -> LinearSystem:
        """Create a high condition number system."""

        # Create ill-conditioned spectrum
        spectrum = self.rng.normal(loc=0.0, scale=1, size=num_cols) ** 2
        spectrum += 1e-6

        A, b, x_opt = self._create_system_from_spectrum(
            num_rows, num_cols, spectrum, prop_range, lev, vt_dis
        )

        return LinearSystem(
            matrix_A=A,
            vector_b=b,
            solution_x=x_opt,
            system_type=SystemType.MID_COND
        )
    
    def _create_high_condition_system(self,
                                    num_rows: int,
                                    num_cols: int,
                                    prop_range: float,
                                    lev: str = "high",
                                    vt_dis: str = "t_dist") -> LinearSystem:
        """Create a high condition number system."""

        alpha = 10
        spectrum = 1.0 / np.arange(1, num_cols + 1)**alpha
        spectrum += 1e-6

        A, b, x_opt = self._create_system_from_spectrum(
            num_rows, num_cols, spectrum, prop_range, lev, vt_dis
        )

        return LinearSystem(
            matrix_A=A,
            vector_b=b,
            solution_x=x_opt,
            system_type=SystemType.HIGH_COND
        )

    def _generate_pd_matrix(self, 
                           size: int,
                           condition_number: float,
                           alpha: float = 1e-3,
                           fix_min: bool = True) -> np.ndarray:
        """Generate a positive definite matrix with specified condition number."""

        B = self.rng.standard_normal(size=(size, size))
        A = B @ B.T
        A += alpha * np.eye(size)
        eigenvalues, Q = np.linalg.eigh(A)

        if fix_min:
            min_eigenvalue = np.min(eigenvalues)
            max_eigenvalue = min_eigenvalue * condition_number
        else:
            max_eigenvalue = np.max(eigenvalues)
            min_eigenvalue = max_eigenvalue / condition_number

        scaling = np.linspace(0, 1, size)
        scaled_eigenvalues = min_eigenvalue + scaling * (max_eigenvalue - min_eigenvalue)
        A_scaled = Q @ np.diag(scaled_eigenvalues) @ Q.T

        return A_scaled

    def _create_system_from_spectrum(self,
                                   num_rows: int,
                                   num_cols: int,
                                   spectrum: np.ndarray,
                                   prop_range: float,
                                   lev: str = "low",
                                   vt_dis: str = "gauss") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create linear system from given singular value spectrum."""

        rank = spectrum.size

        # Create left singular vectors
        if lev == "high":
            U = self._orthonormal_operator_from_mvt(num_rows, rank)
        else:
            U = self._orthonormal_operator(num_rows, rank)

        # Create right singular vectors
        if vt_dis == "gauss":
            Vt = self._orthonormal_operator(rank, num_cols)
        elif vt_dis == "t_dist":
            Vt = self._orthonormal_operator_from_mvt(rank, num_cols)
        elif vt_dis == "ht":
            Vt = self.rng.standard_normal(size=(rank, num_cols))
            row_norms = np.linalg.norm(Vt, axis=1, keepdims=True)
            scaling_factors = self.rng.uniform(0.5, 2.0, size=(rank, 1))
            Vt = (Vt / row_norms) * scaling_factors
        else:
            raise ValueError(f"Unknown Vt distribution: {vt_dis}")

        # Construct matrix A
        A = (U * spectrum) @ Vt

        # Construct right-hand-side with specified proportion in range
        b0 = self.rng.standard_normal(num_rows)
        b_range = U @ (U.T @ b0)
        b_orthog = b0 - b_range
        b_range *= np.mean(spectrum) / la.norm(b_range)
        b_orthog *= np.mean(spectrum) / la.norm(b_orthog)
        b = prop_range * b_range + (1 - prop_range) * b_orthog

        # Compute optimal solution
        x_opt = (Vt.T / spectrum) @ (U.T @ b)

        return A, b, x_opt

    def _orthonormal_operator(self, num_rows: int, num_cols: int) -> np.ndarray:
        """Generate orthonormal matrix using Gaussian random matrix."""
        Q = self.rng.standard_normal(size=(num_rows, num_cols))
        # Q = self.rng.normal(loc=0.0, scale=1, size=(num_rows, num_cols))
        Q, R = la.qr(Q, overwrite_a=True, pivoting=False, mode="economic")
        Q = Q * np.sign(np.diag(R))
        return Q

    def _orthonormal_operator_from_mvt(self, num_rows: int, num_cols: int) -> np.ndarray:
        """Generate orthonormal matrix using multivariate t-distribution."""
        if num_rows < num_cols:
            return self._orthonormal_operator_from_mvt(num_cols, num_rows).T

        Q = self.rng.standard_normal(size=(num_rows, num_cols))
        # c = self.rng.chisquare(df=1, size=(num_rows, 1))
        c = self.rng.chisquare(df=0.1, size=(num_rows, 1)) # more heavy-tailed, to distinguish subampling with leverage score from without leverage score
        Q = Q * np.sqrt(1 / c)
        Q, R = la.qr(Q, overwrite_a=True, pivoting=False, mode="economic")
        Q = Q * np.sign(np.diag(R))
        return Q

class SketchingOperations:
    """Utilities for sketching operations."""

    @staticmethod
    def create_sjlt_operator(num_rows: int,
                           num_cols: int,
                           rng: np.random.Generator,
                           vec_nnz: int = 8) -> spar.csc_matrix:
        """Create Sparse Johnson-Lindenstrauss Transform operator."""

        if num_cols >= num_rows:
            vec_nnz = min(num_cols, vec_nnz)
            row_vecs = []
            bad_size = num_rows < vec_nnz

            if bad_size:
                warnings.warn(
                    f"Can't set {vec_nnz} nonzeros per column for columns of length {num_rows}. "
                    "Sampling indices with replacement instead."
                )

            for i in range(num_cols):
                rows = rng.choice(num_rows, vec_nnz, replace=bad_size)
                row_vecs.append(rows)

            rows = np.concatenate(row_vecs)
            cols = np.repeat(np.arange(num_cols), vec_nnz)

            # Random signs
            vals = np.ones(num_cols * vec_nnz)
            vals[rng.random(num_cols * vec_nnz) <= 0.5] = -1
            vals /= np.sqrt(vec_nnz)

            S = spar.coo_matrix((vals, (rows, cols)), shape=(num_rows, num_cols))
            S = S.tocsc()
        else:
            S = SketchingOperations.create_sjlt_operator(num_cols, num_rows, rng, vec_nnz)
            S = (S.T).tocsr()

        return S

    @staticmethod
    def dim_checks(sampling_factor: float, num_rows: int, num_cols: int) -> int:
        """Check and adjust embedding dimension."""
        assert num_rows >= num_cols
        d = int(sampling_factor * num_cols)

        if d > num_rows:
            warnings.warn(
                f"The embedding dimension d={d} should not be larger than the "
                f"number of rows {num_rows}. Setting d={num_rows}. This will "
                "result in an inefficient algorithm!"
            )
            d = num_rows

        assert d >= num_cols
        return d

class SamplingOperations:
    """Utilities for row sampling operations."""

    @staticmethod
    def compute_leverage_scores(matrix: np.ndarray) -> np.ndarray:
        """Compute leverage scores for matrix rows."""
        leverage_scores = np.sum(matrix ** 2, axis=1)
        return leverage_scores / np.sum(leverage_scores)

    @staticmethod
    def sample_rows(num_rows: int,
                   batch_size: int,
                   probabilities: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        """Sample row indices based on probabilities."""
        return rng.choice(num_rows, batch_size, replace=False, p=probabilities)

    @staticmethod
    def create_sampling_matrix(indices: np.ndarray,
                             num_rows: int,
                             batch_size: int) -> np.ndarray:
        """Create sampling matrix from indices."""
        S_t = np.zeros((batch_size, num_rows), dtype=np.float32)
        row_indices = np.arange(batch_size)
        S_t[row_indices, indices] = 1
        return S_t

    # @staticmethod
    # def create_sampling_matrix(num_rows: int,
    #                            batch_size: int,
    #                            probabilities: np.ndarray,
    #                            rng: np.random.Generator) -> np.ndarray:
    #     """
    #     Sample rows according to the given probability vector and create a reweighted sampling matrix.

    #     Args:
    #         num_rows: Total number of rows in the matrix.
    #         batch_size: Number of rows to sample.
    #         probabilities: Probability distribution over rows (length num_rows).
    #         rng: Random number generator.

    #     Returns:
    #         S_t: Reweighted sampling matrix of shape (batch_size, num_rows)
    #     """
    #     indices = rng.choice(num_rows, batch_size, replace=True, p=probabilities)
    #     S_t = np.zeros((batch_size, num_rows), dtype=np.float32)
    #     row_indices = np.arange(batch_size)
    #     # reweights = 1.0 / np.sqrt(num_rows * probabilities[indices])
    #     # reweights = 1.0 / np.sqrt(batch_size * probabilities[indices])
    #     reweights = 1.0 / np.sqrt(probabilities[indices])
    #     S_t[row_indices, indices] = reweights
    #     return S_t

import numpy as np

from datetime import datetime

from domain.metrics import RewardCalculator
from domain.linear_system import SystemType

from solver.environment import SolverEnvironment

from infrastructure.linear_algebra.data_generation import LinearSystemFactory, SketchingOperations


def get_eval_env(
    random_seed: int = 42,
    system_type: SystemType = SystemType.HIGH_COND,
    num_rows: int = 10000,
    num_cols: int = 50,
    condition_number: float = 2,
    prop_range: float = 1.0,
    lev: str = "high",
    vt_dis: str = "t_dist",
    sampling_factor: int = 8,
    max_iterations: int = 10,
    learning_rate: float = 0.5,
    base_weight: float = 1.0,
    decay_weight: float = 10.0,
    complexity_weight: float = 8.0,
    condition_weight: float = 1.0,
    reward_type: str = "log",
    iterative_sketch: bool = False,
    condition_reward_only: bool = False,
    batch_size: int = 100
) -> SolverEnvironment:
    """Get an evaluation environment.
    
    Args:
        random_seed: Random seed for reproducibility
        system_type: Type of linear system (PSD, LOW_COND, MID_COND, HIGH_COND)
        num_rows: Number of rows in the linear system matrix
        num_cols: Number of columns in the linear system matrix
        condition_number: Condition number for PSD systems
        prop_range: 0-1, proportion of the range of the system to use for the right-hand side
        lev: Leverage score distribution level ("low", "high")
        vt_dis: Distribution type for right singular vectors ("gauss", "t_dist", "ht")
        sampling_factor: Factor for sketch dimension calculation (sketch_dim = sampling_factor * num_cols)
        max_iterations: Maximum iterations for iterative solver
        learning_rate: Learning rate for gradient-based updates
        base_weight: Base weight for reward calculation
        decay_weight: Weight for convergence decay in reward
        complexity_weight: Weight for algorithm complexity penalty in reward
        condition_weight: Weight for condition number penalty in reward
        reward_type: Type of reward function ("log", "exp")
        iterative_sketch: Whether to use iterative sketching in reward calculation
        condition_reward_only: Whether to use only condition number for reward
        batch_size: Batch size for subsampling operations
        
    Returns:
        Configured SolverEnvironment for evaluation
    """

    system_factory = LinearSystemFactory(random_seed=random_seed)

    if system_type == SystemType.PSD:
        linear_system = system_factory.create_system(
            system_type=system_type,
            num_rows=num_rows,
            num_cols=num_cols,
            condition_number=condition_number
        )
    else:  # LOW_COND, MID_COND or HIGH_COND
        linear_system = system_factory.create_system(
            system_type=system_type,
            num_rows=num_rows,
            num_cols=num_cols,
            prop_range=prop_range,
            lev=lev,
            vt_dis=vt_dis
        )

    m, n = linear_system.num_rows, linear_system.num_cols
    rng = np.random.default_rng(random_seed)
    d = SketchingOperations.dim_checks(sampling_factor, m, n)
    sketch_matrix = SketchingOperations.create_sjlt_operator(d, m, rng)

    environment = SolverEnvironment(
        linear_system=linear_system,
        setup_loop=False,
        forward_loop=False,
        update_loop=False,
        max_steps=10000,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        reward_calculator=RewardCalculator(
            base_weight=base_weight,
            decay_weight=decay_weight,
            complexity_weight=complexity_weight,
            condition_weight=condition_weight,
            reward_type=reward_type,
            iterative_sketch=iterative_sketch,
            condition_reward_only=condition_reward_only),
        sketch_matrix=sketch_matrix,
        batch_size=batch_size
    )

    return environment


def time_diff_seconds(t1: str, t2: str) -> float:
    """Compute time difference in seconds between two strings in 'YYYY-MM-DD HH:MM:SS' format."""
    fmt = "%Y-%m-%d %H:%M:%S"
    dt1 = datetime.strptime(t1, fmt)
    dt2 = datetime.strptime(t2, fmt)
    return abs((dt2 - dt1).total_seconds())

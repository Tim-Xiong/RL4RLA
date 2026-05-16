from os import path
from typing import List, Tuple

from notebook.notebook_utils import time_diff_seconds
from domain.operation import OperatorType, OperandType
from domain.linear_system import SystemType
from domain.metrics import RewardCalculator
from solver.variables import AlgorithmState
from solver.environment import SolverEnvironment
from infrastructure.linear_algebra.data_generation import LinearSystemFactory


incomplete_algorithms = set([
    "preconditioner R1",
    "subsampling",
    "sketch and project base",
])

search_paths = {
    "leverage score subsampling": [
        "jacobi iteration",
        "gradient descent",
        "preconditioner R1",
        "preconditioner R1 applying on gradient",
        "sketch and precondition",
        "subsampling",
        "leverage score subsampling",
    ],
    "sketch and solve - low cond": [
        "jacobi iteration",
        "gradient descent",
        "sketch and solve - low cond",
    ],
    "sketch and solve - mid cond": [
        "jacobi iteration",
        "gradient descent",
        "preconditioner R1",
        "preconditioner R1 applying on gradient",
        "sketch and solve - mid cond",
    ],
    "sketch and project": [
        "jacobi iteration",
        "gradient descent",
        "sketch and project base",
        "sketch and project",
    ]
}

algorithms = {
    "jacobi iteration": {
        "setup_loop": [],
        "forward_loop": [
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": []
    },
    # "jacobi iteration equivalent": {
    #     "setup_loop": [],
    #     "forward_loop": [
    #         [OperatorType.VEC_MAT_MUL, OperandType.X_T, OperandType.A, OperandType.V1],
    #         [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
    #     ],
    #     "update_loop": []
    # },
    "gradient descent": {
        "setup_loop": [],
        "forward_loop": [
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1]
        ]
    },
    "preconditioner R1": {
        "setup_loop": [
            [OperatorType.HHQR, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.R1, OperandType.R1, OperandType.R1],
        ],
        "forward_loop": [
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1]
        ]
    },
    "preconditioner R1 applying on gradient": {
        "setup_loop": [
            [OperatorType.HHQR, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.R1, OperandType.R1, OperandType.R1],
        ],
        "forward_loop": [
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1],
            [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1]
        ]
    },
    "sketch and precondition": {
        "setup_loop": [
            [OperatorType.HHQR, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.SKETCH, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.HHQR, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.R1, OperandType.R1, OperandType.R1],
        ],
        "forward_loop": [
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1],
            [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1]
        ]
    },
    "subsampling": {
        "setup_loop": [
            [OperatorType.HHQR, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.SKETCH, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.HHQR, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.R1, OperandType.R1, OperandType.R1],
        ],
        "forward_loop": [
            [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1],
            [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1]
        ]
    },
    "leverage score subsampling": {
        "setup_loop": [
            [OperatorType.LEVERAGE_SCORE, OperandType.A, OperandType.NONE, OperandType.V3],
            [OperatorType.HHQR, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.SKETCH, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.HHQR, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.R1, OperandType.R1, OperandType.R1],
        ],
        "forward_loop": [
            [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1],
            [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1]
        ]
    },
    "sketch and solve - low cond": {
        "setup_loop": [
            [OperatorType.LEVERAGE_SCORE, OperandType.A, OperandType.NONE, OperandType.V3],
            [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE]
        ],
        "forward_loop": [
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1]
        ]
    },
    "sketch and solve - mid cond": {
        "setup_loop": [
            [OperatorType.LEVERAGE_SCORE, OperandType.A, OperandType.NONE, OperandType.V3],
            [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
            [OperatorType.HHQR, OperandType.A, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.R1, OperandType.R1, OperandType.R1],
        ],
        "forward_loop": [
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1],
            [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1]
        ]
    },
    "sketch and project base": {
        "setup_loop": [],
        "forward_loop": [
            [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1]
        ]
    },
    "sketch and project": {
        "setup_loop": [],
        "forward_loop": [
            [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.A, OperandType.A, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.MAT_VEC_MUL, OperandType.R1, OperandType.V1, OperandType.V1],
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1]
        ]
    },
    "sketch and project equivalent": {
        "setup_loop": [],
        "forward_loop": [
            [OperatorType.SUBSAMPLING, OperandType.V3, OperandType.NONE, OperandType.NONE],
            [OperatorType.MAT_VEC_MUL, OperandType.A, OperandType.X_T, OperandType.V1],
            [OperatorType.VEC_VEC_SUB, OperandType.V1, OperandType.B, OperandType.V1],
        ],
        "update_loop": [
            [OperatorType.MAT_MAT_TRANS_MUL, OperandType.A, OperandType.A, OperandType.R1],
            [OperatorType.MAT_INV, OperandType.R1, OperandType.NONE, OperandType.R1],
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.R1, OperandType.V1],
            [OperatorType.VEC_MAT_MUL, OperandType.V1, OperandType.A, OperandType.V1]
        ]
    }
}

def list_algorithm_names():
    """List all algorithm names."""
    print(list(algorithms.keys()))

def list_search_path_names():
    """List all algorithm search paths."""
    print(list(search_paths.keys()))

def get_sample_algorithms() -> List[Tuple[str, AlgorithmState]]:
    """Get a list of sample name-algorithm tuples."""
    sample_algorithms = []
    for algorithm_name, algorithm in algorithms.items():
        sample_algorithms.append((
            algorithm_name,
            AlgorithmState.from_dict(algorithm)
        ))
    return sample_algorithms

def get_sample_algorithm_by_name(name: str) -> AlgorithmState:
    """Get a sample algorithm by name."""
    return AlgorithmState.from_dict(algorithms[name])

def get_algorithms_search_paths() -> dict[str, List[AlgorithmState]]:
    """Get a dictionary of algorithm search paths."""
    algorithms_search_paths = {}
    for path_name, algorithm_names in search_paths.items():
        algorithms_search_paths[path_name] = [AlgorithmState.from_dict(algorithms[name]) for name in algorithm_names]
    return algorithms_search_paths

def get_algorithms_search_path_by_name(name: str) -> List[AlgorithmState]:
    """Get an algorithm search path by name."""
    path = search_paths[name]
    return [AlgorithmState.from_dict(algorithms[name]) for name in path]

def get_mcgs_ucd_search_paths(complete_only: bool = False) -> dict[str, List[Tuple[float, AlgorithmState]]]:
    """Get a dictionary of MCGS UCD search paths with time taken."""
    algorithms_search_paths = {}
    for path_name, path in mcgs_ucd_search_paths.items():
        search_path = []
        accumulated_time = 0.0
        for time_taken, algorithm_name in path:
            if complete_only and algorithm_name in incomplete_algorithms:
                if time_taken is not None:
                    accumulated_time += time_taken
                continue
            total_time = None
            if time_taken is not None:
                total_time = time_taken + accumulated_time
                accumulated_time = 0.0
            elif accumulated_time > 0.0:
                total_time = accumulated_time
                accumulated_time = 0.0
            search_path.append((total_time, AlgorithmState.from_dict(algorithms[algorithm_name])))
        algorithms_search_paths[path_name] = search_path
    return algorithms_search_paths

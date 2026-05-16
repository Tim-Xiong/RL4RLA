import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
import argparse
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

from mcgs.search import MCGSPlayer
from mcts.policies import UniformPolicy

from domain.metrics import RewardCalculator
from domain.linear_system import SystemType
from domain.operation import OperatorType, OperandType

from solver.environment import SolverEnvironment
from solver.variables import AlgorithmState

from infrastructure.linear_algebra.data_generation import LinearSystemFactory, SketchingOperations
from infrastructure.logging.experiment_db import ExperimentDatabase
from notebook.sample_algorithms import algorithms as sample_algorithms


class TeeOutput:
    """Redirect output to both console and file."""
    def __init__(self, *files):
        self.files = files

    def write(self, txt):
        for file_obj in self.files:
            file_obj.write(txt)
            file_obj.flush()

    def flush(self):
        for file_obj in self.files:
            file_obj.flush()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MCGS (Monte Carlo Graph Search) for algorithm discovery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Algorithm configuration
    parser.add_argument('--initial-algorithm', type=str, default=None,
                        help='Path to JSON file containing initial algorithm')
    parser.add_argument('--default-algorithm', type=str, default=None,
                        help='Name of predefined algorithm from sample_algorithms.py')
    parser.add_argument('--new-steps', type=int, default=2,
                        help='Number of new operations to add (search depth, max ~3)')

    # Search configuration
    parser.add_argument('--n-playouts', type=int, default=200,
                        help='Number of MCGS playouts')
    parser.add_argument('--c', type=float, default=0.5,
                        help='Exploration constant for PUCT/UCT/UCD')
    parser.add_argument('--score-fn', type=str, default='ucd',
                        choices=['puct', 'uct', 'ucd'],
                        help='Score function for node selection')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for final action selection (0.0 = greedy)')
    parser.add_argument('--graph-reuse', action='store_true', default=False,
                        help='Reuse graph structure across searches')
    parser.add_argument('--early-stopping', action='store_true', default=False,
                        help='Enable early stopping when top action is clearly better')
    parser.add_argument('--stopping-factor', type=float, default=1.0,
                        help='Stopping factor for early stopping criterion')
    parser.add_argument('--use-lucb', action='store_true', default=False,
                        help='Use LUCB (Lower and Upper Confidence Bound) for root action selection')
    parser.add_argument('--best-action-metric', type=str, default='visits',
                        choices=['visits', 'q_value'],
                        help='Metric for selecting best action (visits or q_value)')

    # Environment configuration
    parser.add_argument('--system-type', type=str, default='PSD',
                        choices=['PSD', 'LOW_COND', 'MID_COND', 'HIGH_COND'],
                        help='Linear system type')
    parser.add_argument('--num-rows', type=int, default=5,
                        help='Number of rows in linear system')
    parser.add_argument('--num-cols', type=int, default=5,
                        help='Number of columns in linear system')
    parser.add_argument('--condition-number', type=float, default=2.0,
                        help='Condition number for PSD systems')
    parser.add_argument('--prop-range', type=float, default=1.0,
                        help='Range of singular values for non-PSD systems')
    parser.add_argument('--lev', type=str, default='high',
                        choices=['low', 'high'],
                        help='Leverage score distribution for non-PSD systems')
    parser.add_argument('--vt-dis', type=str, default='t_dist',
                        choices=['gauss', 't_dist', 'ht'],
                        help='Right singular vector distribution for non-PSD systems')

    # Loop configuration
    parser.add_argument('--setup-loop', action='store_true', default=False,
                        help='Allow modifying setup loop')
    parser.add_argument('--forward-loop', action='store_true', default=False,
                        help='Allow modifying forward loop')
    parser.add_argument('--update-loop', action='store_true', default=False,
                        help='Allow modifying update loop')

    # Solver configuration
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum iterations for solver')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate for solver')
    parser.add_argument('--sampling-factor', type=int, default=8,
                        help='Sampling factor for sketch matrix')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for evaluation')

    # Reward configuration
    parser.add_argument('--base-weight', type=float, default=1.0,
                        help='Weight for base convergence reward')
    parser.add_argument('--decay-weight', type=float, default=10.0,
                        help='Weight for decay penalty')
    parser.add_argument('--complexity-weight', type=float, default=8.0,
                        help='Weight for complexity penalty')
    parser.add_argument('--condition-weight', type=float, default=1.0,
                        help='Weight for condition number penalty')
    parser.add_argument('--reward-type', type=str, default='log',
                        help='Reward calculation type')
    parser.add_argument('--iterative-sketch', action='store_true', default=False,
                        help='Use iterative sketching')
    parser.add_argument('--condition-reward-only', action='store_true', default=False,
                        help='Focus exclusively on conditioning metrics')

    # Output configuration
    parser.add_argument('--log-dir', type=str, default='logs/mcgs',
                        help='Directory for logs')
    parser.add_argument('--db-path', type=str, default='logs/experiments.db',
                        help='Path to experiment database')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Experiment tracking
    parser.add_argument('--target-algorithm', type=str, default=None,
                        help='Name of target algorithm from sample_algorithms.py to check if found')
    parser.add_argument('--focus-operator', type=str, default=None,
                        help='Comma-separated list of operators to focus on in statistics (e.g., "LEVERAGE_SCORE,SUBSAMPLING")')

    return parser.parse_args()


def load_algorithm_from_json(json_path):
    """Load initial algorithm from JSON file (for --initial-algorithm argument)."""
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'algorithm' in data:
        algo_dict = data['algorithm']
    else:
        raise ValueError(f"Algorithm not found in JSON file: {json_path}")

    # Convert enum names back to enum objects
    for loop_name in ['setup_loop', 'forward_loop', 'update_loop']:
        if loop_name in algo_dict:
            algo_dict[loop_name] = [
                [OperatorType[op[0]], OperandType[op[1]], OperandType[op[2]], OperandType[op[3]]]
                for op in algo_dict[loop_name]
            ]

    return AlgorithmState.from_dict(algo_dict)


# Example usage:
# python run_mcgs.py --forward-loop --log-dir logs/mcgs/ --db-path logs/experiments.db
def main():
    args = parse_args()

    # Validate and parse target algorithm
    target_algorithm_state = None
    if args.target_algorithm:
        if args.target_algorithm not in sample_algorithms:
            raise ValueError(f"Unknown target algorithm: {args.target_algorithm}. "
                           f"Available: {list(sample_algorithms.keys())}")
        target_algorithm_state = AlgorithmState.from_dict(sample_algorithms[args.target_algorithm])

    # Validate and parse focus operators
    focus_operators = None
    if args.focus_operator:
        try:
            focus_operators = [OperatorType[op.strip()] for op in args.focus_operator.split(',')]
        except KeyError as e:
            raise ValueError(f"Invalid operator name in focus-operator: {e}. "
                           f"Available: {[op.name for op in OperatorType]}")

    # Setup output directory and logging
    logs_dir = Path(args.log_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]  # Include milliseconds
    log_file = logs_dir / f"mcgs_{timestamp}.log"

    # Initialize database
    db = ExperimentDatabase(args.db_path)

    # Redirect stdout to both console and file
    with open(log_file, 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(original_stdout, f)

        print(f"=== MCGS Algorithm Discovery - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        print(f"Output logged to: {log_file}")
        print(f"Database: {args.db_path}")
        print(f"System: {args.system_type}, New steps: {args.new_steps}, Playouts: {args.n_playouts}")
        if args.target_algorithm:
            print(f"Target algorithm: {args.target_algorithm}")
        print("=" * 80)

        # Create or load initial algorithm
        if args.initial_algorithm:
            print(f"Loading initial algorithm from: {args.initial_algorithm}")
            initial_algorithm = load_algorithm_from_json(args.initial_algorithm)
            print(f"Loaded algorithm with {initial_algorithm.step} steps")
        elif args.default_algorithm:
            if args.default_algorithm not in sample_algorithms:
                raise ValueError(f"Unknown algorithm: {args.default_algorithm}. "
                               f"Available: {list(sample_algorithms.keys())}")
            print(f"Using predefined algorithm: {args.default_algorithm}")
            initial_algorithm = AlgorithmState.from_dict(sample_algorithms[args.default_algorithm])
            print(f"Loaded algorithm with {initial_algorithm.step} steps")
        else:
            print("Creating empty initial algorithm")
            initial_algorithm = AlgorithmState.from_dict({
                "setup_loop": [],
                "forward_loop": [],
                "update_loop": []
            })
        print(initial_algorithm.algorithm.get_readable_representation())

        max_steps = args.new_steps + initial_algorithm.step

        # Create linear system
        print(f"\nCreating {args.system_type} linear system...")
        system_factory = LinearSystemFactory(random_seed=args.random_seed)

        system_type = SystemType[args.system_type]
        if system_type == SystemType.PSD:
            linear_system = system_factory.create_system(
                system_type=system_type,
                num_rows=args.num_rows,
                num_cols=args.num_cols,
                condition_number=args.condition_number
            )
        else:
            linear_system = system_factory.create_system(
                system_type=system_type,
                num_rows=args.num_rows,
                num_cols=args.num_cols,
                prop_range=args.prop_range,
                lev=args.lev,
                vt_dis=args.vt_dis
            )
        print(f"System shape: ({linear_system.num_rows}, {linear_system.num_cols})")

        # Create sketch matrix
        print("Creating sketch matrix...")
        m, n = linear_system.num_rows, linear_system.num_cols
        rng = np.random.default_rng(args.random_seed)
        d = SketchingOperations.dim_checks(args.sampling_factor, m, n)
        sketch_matrix = SketchingOperations.create_sjlt_operator(d, m, rng)
        print(f"Sketch dimension: {d}")

        # Create environment
        print("\nCreating solver environment...")
        environment = SolverEnvironment(
            linear_system=linear_system,
            setup_loop=args.setup_loop,
            forward_loop=args.forward_loop,
            update_loop=args.update_loop,
            max_steps=max_steps,
            max_iterations=args.max_iterations,
            learning_rate=args.learning_rate,
            reward_calculator=RewardCalculator(
                base_weight=args.base_weight,
                decay_weight=args.decay_weight,
                complexity_weight=args.complexity_weight,
                condition_weight=args.condition_weight,
                reward_type=args.reward_type,
                iterative_sketch=args.iterative_sketch,
                condition_reward_only=args.condition_reward_only
            ),
            sketch_matrix=sketch_matrix,
            batch_size=args.batch_size,
            random_seed=args.random_seed
        )

        # Create MCGS player
        print("Creating MCGS player...")
        player = MCGSPlayer(
            random_seed=args.random_seed,
            initial_algorithm=initial_algorithm,
            policy_strategy=UniformPolicy(),
            c=args.c,
            score_fn=args.score_fn,
            n_playouts=args.n_playouts,
            add_exploration_noise=False,
            graph_reuse=args.graph_reuse,
            target_algorithm=target_algorithm_state,
            focus_operators=focus_operators,
            early_stopping=args.early_stopping,
            stopping_factor=args.stopping_factor,
            use_lucb=args.use_lucb,
            best_action_metric=args.best_action_metric
        )

        # Create experiment record in database
        experiment_id = None
        try:
            print("Creating experiment record in database...")
            experiment_name = f"mcgs_{args.score_fn}_{timestamp}"
            config = vars(args)
            experiment_id = db.create_experiment(
                name=experiment_name,
                search_method="mcgs",
                config=config,
                notes=f"Log file: {log_file}"
            )
            print(f"Experiment ID: {experiment_id}")

            # Run MCGS search
            print(f"\nStarting MCGS search...")
            print("-" * 50)

            search_start = time.perf_counter()
            final_algorithm = player.run_search(
                environment,
                new_steps=args.new_steps,
                temperature=args.temperature,
                experiment_db=db,
                experiment_id=experiment_id
            )
            search_time = time.perf_counter() - search_start

            print("-" * 50)
            print(f"Search completed in {search_time:.2f} seconds")
            print(f"\nFinal algorithm:")
            print(final_algorithm.algorithm.get_readable_representation())

            # Get results
            reward_metrics = environment.get_reward_metrics(final_algorithm)
            print(f"\nReward metrics:\n{reward_metrics}")

            reward = environment.get_reward(final_algorithm)
            print(f"Final reward: {reward}")

            # Get search statistics
            search_stats = player.searcher.get_search_statistics()
            print("\nMCGS Search Statistics:")
            for key, value in search_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            player.searcher.print_cache_info(environment)

            # Check if target algorithm was found
            target_found = None
            if target_algorithm_state is not None:
                target_found = (final_algorithm == target_algorithm_state)
                print(f"\nTarget algorithm '{args.target_algorithm}' found: {target_found}")

            # Update database with final results
            final_alg_dict = final_algorithm.to_dict()
            db.update_experiment_result(experiment_id, final_alg_dict, reward)

            print("=" * 80)
            print(f"=== MCGS Algorithm Discovery - Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            print(f"Experiment ID: {experiment_id}")
            print(f"Database: {args.db_path}")
            print(f"Final reward: {reward:.4f}")

        except Exception as e:
            print(f"Error during experiment: {e}")
            # Mark experiment as failed (if it was created)
            if experiment_id is not None:
                empty_alg = AlgorithmState.from_dict({"setup_loop": [], "forward_loop": [], "update_loop": []})
                db.update_experiment_result(experiment_id, empty_alg.to_dict(), float('-inf'))
            raise
        finally:
            db.close()
            sys.stdout = original_stdout

    print(f"Experiment completed! Logs saved to: {log_file}")


if __name__ == "__main__":
    main()

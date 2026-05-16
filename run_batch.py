#!/usr/bin/env python3
"""
Batch experiment runner for MCGS/MCTS experiments.

Usage:
    # Run all experiments in a directory
    python run_batch.py hp/mcgs_ucd/

    # Run specific config files
    python run_batch.py hp/mcgs_ucd/leverage_score_subsampling/0.yml hp/mcgs_ucd/leverage_score_subsampling/1.yml

    # Run with parallelism (e.g., 4 experiments at once)
    python run_batch.py hp/mcgs_ucd/ --parallel 4

    # Dry run to see what would be executed
    python run_batch.py hp/mcgs_ucd/ --dry-run

    # Filter by pattern
    python run_batch.py hp/ --pattern "mcgs_ucd/*/0.yml"

    # Retry failed experiments or experiments that didn't find target
    python run_batch.py hp/mcgs_ucd/ --retry 3

    # Combine repeat and retry (up to repeat * retry total experiments per config)
    python run_batch.py hp/mcgs_ucd/ --repeat 5 --retry 3

    # Override random seed (repeated experiments use incremental seeds: seed, seed+1, seed+2, ...)
    python run_batch.py hp/mcgs_ucd/ --repeat 3 --random-seed 42
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import yaml
import fnmatch


def find_config_files(paths, pattern=None):
    """Find all YAML config files in given paths, skipping hp/dummy/."""
    config_files = []

    for path in paths:
        path = Path(path)

        if path.is_file() and path.suffix in ['.yml', '.yaml']:
            # Skip file if it is in hp/dummy/
            if "hp/dummy/" in str(path).replace("\\", "/"):
                continue
            config_files.append(path)
        elif path.is_dir():
            # Recursively find all YAML files, skipping those in hp/dummy/
            for yaml_file in sorted(path.rglob('*.yml')):
                if "hp/dummy/" in str(yaml_file).replace("\\", "/"):
                    continue
                config_files.append(yaml_file)
            for yaml_file in sorted(path.rglob('*.yaml')):
                if "hp/dummy/" in str(yaml_file).replace("\\", "/"):
                    continue
                config_files.append(yaml_file)

    # Apply pattern filter if specified
    if pattern:
        config_files = [
            f for f in config_files
            if fnmatch.fnmatch(str(f), f"*{pattern}*")
        ]

    return sorted(set(config_files))


def load_yaml_config(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def yaml_to_args(config, yaml_path, seed_override=None):
    """Convert YAML config to command-line arguments."""
    args = []

    # Handle repeat: use yaml_path as tuple (path, repeat_idx) if needed
    repeat_idx = None
    if isinstance(yaml_path, tuple):
        yaml_path, repeat_idx = yaml_path
    yaml_path_obj = Path(yaml_path)
    # Auto-set log-dir based on yaml path (hp/ -> logs/)
    log_dir = str(yaml_path_obj.parent).replace('hp/', 'logs/', 1).replace('hp\\', 'logs\\', 1)
    log_dir = f"{log_dir}/{yaml_path_obj.stem}/"

    if 'log-dir' not in config or config['log-dir'] is None:
        config['log-dir'] = log_dir

    # Handle random seed override and incremental seeds for repeated runs
    if seed_override is not None:
        config = config.copy()  # Don't modify original
        config['random-seed'] = seed_override
    
    # Increment random seed for repeated/retried runs
    if repeat_idx is not None and 'random-seed' in config:
        config = config.copy()  # Don't modify original
        base_seed = config['random-seed']
        # Extract numeric part from repeat_idx (handles "0", "retry1", "0_retry1", etc.)
        if isinstance(repeat_idx, str):
            # Parse numeric part from strings like "0_retry1" or "retry1"
            numeric_part = 0
            if '_retry' in repeat_idx:
                parts = repeat_idx.split('_retry')
                numeric_part = int(parts[0]) if parts[0].isdigit() else 0
                numeric_part += int(parts[1]) * 1000  # Offset retries by 1000
            elif 'retry' in repeat_idx:
                numeric_part = int(repeat_idx.replace('retry', '')) * 1000
            else:
                numeric_part = int(repeat_idx) if repeat_idx.isdigit() else 0
            config['random-seed'] = base_seed + numeric_part
        else:
            config['random-seed'] = base_seed + repeat_idx

    # Map YAML keys to command-line arguments
    for key, value in config.items():
        if value is None or value == 'null':
            continue

        # Skip 'script' key - it's for run_batch.py only
        if key == 'script':
            continue

        # Convert underscores to hyphens for CLI args
        arg_name = f"--{key}"

        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
            elif key == 'graph-reuse' and not value:
                args.append('--no-graph-reuse')
        else:
            args.extend([arg_name, str(value)])

    return args


def determine_script(yaml_path):
    """Determine which run script to use based on path or config."""
    path_str = str(yaml_path)

    # Check path structure
    if '/mcts/' in path_str and '/mcgs' not in path_str:
        return 'run_mcts.py'
    elif '/mcgs' in path_str and '/mcts' not in path_str:
        return 'run_mcgs.py'

    raise ValueError(f"Can't determine script automatically for path: {path_str}")


def check_target_found(config, db_path):
    """Check if experiment found target algorithm by querying database."""
    from infrastructure.logging.experiment_db import ExperimentDatabase
    from solver.variables import AlgorithmState
    from notebook.sample_algorithms import algorithms as sample_algorithms

    target_name = config.get('target-algorithm')
    if not target_name or target_name not in sample_algorithms:
        return None

    # Get latest experiment from database
    try:
        db = ExperimentDatabase(db_path)
        experiments = db.list_experiments(limit=1)
        if not experiments:
            db.close()
            return None

        latest_exp = db.get_experiment(experiments[0]['id'])
        db.close()

        if not latest_exp or not latest_exp['final_algorithm']:
            return None

        # Compare algorithms
        target_algo = AlgorithmState.from_dict(sample_algorithms[target_name])
        final_algo = AlgorithmState.from_json(latest_exp['final_algorithm'])

        return target_algo.algorithm == final_algo.algorithm
    except Exception as e:
        import traceback
        print(f"Error checking target found: {e}")
        print(traceback.format_exc())
        return None


def run_experiment(yaml_path, dry_run=False, verbose=False, script_override=None, seed_override=None):
    """Run a single experiment from a YAML config file."""
    try:
        # Handle repeat: yaml_path may be tuple (path, repeat_idx)
        actual_path = yaml_path[0] if isinstance(yaml_path, tuple) else yaml_path
        config = load_yaml_config(actual_path)
        
        # Determine script: CLI override > YAML config > auto-detect from path
        if script_override:
            script = script_override
        elif 'script' in config and config['script']:
            script = config['script']
        else:
            script = determine_script(actual_path)
        
        args = yaml_to_args(config, yaml_path, seed_override)

        cmd = ['python', script] + args

        if dry_run or verbose:
            display_path = actual_path if isinstance(yaml_path, tuple) else yaml_path
            repeat_info = f" (run {yaml_path[1]})" if isinstance(yaml_path, tuple) else ""
            print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {display_path}{repeat_info}")
            # Quote arguments with spaces for display
            display_cmd = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
            print(f"  Command: {display_cmd}")

        if dry_run:
            return yaml_path, 0, "Dry run - not executed", None

        # Run the experiment
        start_time = datetime.now()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        target_found = None
        if result.returncode == 0:
            status = f"SUCCESS ({duration:.1f}s)"

            # Check if target algorithm was found
            db_path = next((args[i+1] for i, arg in enumerate(args) if arg == '--db-path'), 'logs/experiments.db')
            target_found = check_target_found(config, db_path)
            if target_found is True:
                status += " - Target FOUND"
            elif target_found is False:
                status += " - Target NOT found"
        else:
            status = f"FAILED ({duration:.1f}s)"
            if verbose:
                print(f"  STDERR: {result.stderr[:500]}")

        return yaml_path, result.returncode, status, target_found

    except Exception as e:
        return yaml_path, -1, f"ERROR: {str(e)}", None


def run_experiment_with_retry(yaml_path, max_retries=1, dry_run=False, verbose=False, script_override=None, seed_override=None):
    """Run experiment with retry logic on failure or when target not found."""
    for attempt in range(max_retries):
        # For retries after the first attempt, mutate the yaml_path to include a retry seed
        current_yaml_path = yaml_path
        if attempt > 0:
            # Create a new tuple with retry marker to force seed increment
            if isinstance(yaml_path, tuple):
                # yaml_path is (path, repeat_idx), add retry_idx
                current_yaml_path = (yaml_path[0], f"{yaml_path[1]}_retry{attempt}")
            else:
                # yaml_path is just path, add retry_idx
                current_yaml_path = (yaml_path, f"retry{attempt}")
        
        _, returncode, status, target_found = run_experiment(
            current_yaml_path, dry_run, verbose, script_override, seed_override
        )
        
        # Success conditions: returncode == 0 AND (target_found is None or True)
        # Retry conditions: returncode != 0 OR target_found == False
        should_retry = (returncode != 0 or target_found is False) and attempt < max_retries - 1
        
        if should_retry:
            if verbose or True:  # Always show retry attempts
                display_path = yaml_path[0] if isinstance(yaml_path, tuple) else yaml_path
                reason = "failed" if returncode != 0 else "target not found"
                print(f"  Retrying {display_path} (attempt {attempt + 2}/{max_retries}) - {reason}")
        else:
            # Return the original yaml_path, not the mutated one
            return yaml_path, returncode, status, target_found
    
    # Should not reach here but return last result just in case
    return yaml_path, returncode, status, target_found


def main():
    parser = argparse.ArgumentParser(
        description='Run batch experiments from YAML config files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('paths', nargs='+',
                       help='YAML config files or directories containing them')
    parser.add_argument('--parallel', '-p', type=int, default=1,
                       help='Number of experiments to run in parallel (default: 1)')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Filter config files by pattern (e.g., "mcgs_ucd/*/0.yml")')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be executed without running')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--continue-on-error', action='store_true', default=True,
                       help='Continue running experiments even if some fail')
    parser.add_argument('--repeat', type=int, default=1,
                       help='Number of times to repeat each experiment')
    parser.add_argument('--retry', type=int, default=1,
                       help='Number of times to retry failed experiments or experiments that did not find target (default: 1, no retry)')
    parser.add_argument('--script', type=str, default=None,
                       help='Script to run (e.g., run_mcgs.py, run_mcts.py). Auto-detected if not specified.')
    parser.add_argument('--random-seed', type=int, default=None,
                       help='Override random seed from YAML config. Repeated experiments will use incremental seeds (seed, seed+1, seed+2, ...)')

    args = parser.parse_args()

    # Find all config files
    config_files = find_config_files(args.paths, args.pattern)

    if not config_files:
        print("No config files found!")
        return 1

    # Create repeated experiments if --repeat > 1
    if args.repeat >= 1:
        repeated_configs = []
        for cf in config_files:
            for i in range(args.repeat):
                repeated_configs.append((cf, i))
        config_files = repeated_configs
        print(f"Found {len(config_files) // args.repeat} config file(s), repeating {args.repeat} times each = {len(config_files)} total experiments")
    else:
        print(f"Found {len(config_files)} config file(s)")
    if args.dry_run:
        print("[DRY RUN MODE - no experiments will be executed]")
    print("=" * 80)

    # Run experiments
    results = []
    failed_count = 0

    if args.parallel > 1 and not args.dry_run:
        # Parallel execution
        print(f"Running experiments with parallelism={args.parallel}")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(run_experiment_with_retry, cf, args.retry, args.dry_run, args.verbose, args.script, args.random_seed): cf
                for cf in config_files
            }

            for future in as_completed(futures):
                yaml_path, returncode, status, _ = future.result()
                results.append((yaml_path, returncode, status))

                display_path = yaml_path[0] if isinstance(yaml_path, tuple) else yaml_path
                repeat_info = f" (run {yaml_path[1]})" if isinstance(yaml_path, tuple) else ""
                print(f"[{len(results)}/{len(config_files)}] {display_path}{repeat_info}: {status}")

                if returncode != 0:
                    failed_count += 1
                    if not args.continue_on_error:
                        print("Stopping due to failure (use --continue-on-error to continue)")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        # Sequential execution
        for i, config_file in enumerate(config_files, 1):
            yaml_path, returncode, status, _ = run_experiment_with_retry(
                config_file, args.retry, args.dry_run, args.verbose, args.script, args.random_seed
            )
            results.append((yaml_path, returncode, status))

            display_path = yaml_path[0] if isinstance(yaml_path, tuple) else yaml_path
            repeat_info = f" (run {yaml_path[1]})" if isinstance(yaml_path, tuple) else ""
            print(f"[{i}/{len(config_files)}] {display_path}{repeat_info}: {status}")

            if returncode != 0:
                failed_count += 1
                if not args.continue_on_error and not args.dry_run:
                    print("Stopping due to failure (use --continue-on-error to continue)")
                    break

    # Summary
    print("=" * 80)
    print(f"\nBatch Execution Summary:")
    print(f"  Total experiments: {len(config_files)}")
    print(f"  Completed: {len(results)}")
    print(f"  Successful: {len(results) - failed_count}")
    print(f"  Failed: {failed_count}")

    if failed_count > 0 and not args.dry_run:
        print("\nFailed experiments:")
        for yaml_path, returncode, status in results:
            if returncode != 0:
                display_path = yaml_path[0] if isinstance(yaml_path, tuple) else yaml_path
                repeat_info = f" (run {yaml_path[1]})" if isinstance(yaml_path, tuple) else ""
                print(f"  - {display_path}{repeat_info}: {status}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

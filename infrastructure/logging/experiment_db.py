"""Database interface for experiment tracking and replay."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import logging


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.name
        return super().default(obj)


class ExperimentDatabase:
    """
    SQLite database interface for tracking MCGS/MCTS/FunSearch experiments.

    Minimal schema design:
    - experiments: stores config, seed, initial algorithm, and results
    - playouts: stores only (experiment_id, step_idx, playout_idx, reward)

    This enables deterministic replay of entire search process from stored rewards.
    """

    def __init__(self, db_path: str, buffer_size: int = 100):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            buffer_size: Number of playouts to buffer before flushing to DB
        """
        # Check if database file exists, else will be created
        # if not Path(db_path).exists():
        #     raise FileNotFoundError(f"Database file {db_path} not found")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

        # Performance optimizations
        self.conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        self.conn.execute("PRAGMA foreign_keys=ON")   # Enforce foreign keys
        self.conn.execute("PRAGMA synchronous=NORMAL") # Balanced safety/speed

        # Initialize schema
        self._initialize_schema()

        # Buffering for efficient batch inserts
        self.buffer_size = buffer_size
        self.playout_buffer: List[Tuple] = []

        self.logger = logging.getLogger(__name__)

    def _initialize_schema(self):
        """Create tables and indexes if they don't exist."""
        schema_path = Path(__file__).parent / "experiment_db_schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Execute each statement separately
        for statement in schema_sql.split(';'):
            statement = statement.strip()
            if statement:
                self.conn.execute(statement)

        self.conn.commit()

    def create_experiment(
        self,
        name: str,
        search_method: str,
        config: Dict[str, Any],
        notes: Optional[str] = None
    ) -> int:
        """
        Create a new experiment record.

        Args:
            name: Unique experiment name
            search_method: "mcgs", "mcts", or "funsearch"
            config: Complete configuration including:
                - Search hyperparameters (c, score_fn, n_playouts, etc.)
                - system_config: Linear system parameters
                - initial_algorithm: Starting algorithm state
                - random_seed: For deterministic replay
            notes: Optional notes about the experiment

        Returns:
            experiment_id: The database ID of the created experiment
        """
        cursor = self.conn.execute(
            """INSERT INTO experiments
               (name, search_method, config, started_at, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (name, search_method, json.dumps(config), datetime.now(), notes)
        )
        self.conn.commit()

        experiment_id = cursor.lastrowid
        self.logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id

    def log_playout(
        self,
        experiment_id: int,
        step_idx: int,
        playout_idx: int,
        reward: float
    ):
        """
        Log a single playout (buffered for performance).

        Args:
            experiment_id: ID of the experiment
            step_idx: Which step in multi-step search (0 to new_steps-1)
            playout_idx: Which playout within this step (0 to n_playouts-1)
            reward: The reward obtained from this playout
        """
        self.playout_buffer.append((
            experiment_id,
            step_idx,
            playout_idx,
            reward,
            datetime.now()
        ))

        # Flush buffer if full
        if len(self.playout_buffer) >= self.buffer_size:
            self._flush_playout_buffer()

    def _flush_playout_buffer(self):
        """Flush buffered playouts to database in a single transaction."""
        if not self.playout_buffer:
            return

        with self.conn:  # Automatic transaction
            self.conn.executemany(
                """INSERT INTO playouts
                   (experiment_id, step_idx, playout_idx, reward, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                self.playout_buffer
            )

        self.logger.debug(f"Flushed {len(self.playout_buffer)} playouts to database")
        self.playout_buffer = []

    def update_experiment_result(
        self,
        experiment_id: int,
        final_algorithm: Dict[str, Any],
        final_reward: float
    ):
        """
        Update experiment with final result.

        Args:
            experiment_id: ID of the experiment
            final_algorithm: The discovered algorithm
            final_reward: Its reward
        """
        self._flush_playout_buffer()  # Ensure all playouts are saved

        self.conn.execute(
            """UPDATE experiments
               SET completed_at = ?, final_algorithm = ?, final_reward = ?
               WHERE id = ?""",
            (datetime.now(), json.dumps(final_algorithm, cls=EnumEncoder), final_reward, experiment_id)
        )
        self.conn.commit()
        self.logger.info(f"Updated experiment {experiment_id} with final result (reward: {final_reward:.4f})")

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve experiment by ID.

        Returns:
            Dictionary with experiment data or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "search_method": row["search_method"],
            "config": json.loads(row["config"]),
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "final_algorithm": row["final_algorithm"],
            "final_reward": row["final_reward"],
            "notes": row["notes"]
        }

    def get_experiment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment by name."""
        cursor = self.conn.execute(
            "SELECT * FROM experiments WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "name": row["name"],
            "search_method": row["search_method"],
            "config": json.loads(row["config"]),
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "final_algorithm": json.loads(row["final_algorithm"]) if row["final_algorithm"] else None,
            "final_reward": row["final_reward"],
            "notes": row["notes"]
        }

    def get_playouts(
        self,
        experiment_id: int,
        step_idx: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get playouts for an experiment, optionally filtered by step.

        Args:
            experiment_id: ID of the experiment
            step_idx: If provided, only return playouts for this step

        Returns:
            List of playout dictionaries, ordered by step_idx and playout_idx
        """
        if step_idx is not None:
            cursor = self.conn.execute(
                """SELECT * FROM playouts
                   WHERE experiment_id = ? AND step_idx = ?
                   ORDER BY playout_idx""",
                (experiment_id, step_idx)
            )
        else:
            cursor = self.conn.execute(
                """SELECT * FROM playouts
                   WHERE experiment_id = ?
                   ORDER BY step_idx, playout_idx""",
                (experiment_id,)
            )

        return [
            {
                "id": row["id"],
                "experiment_id": row["experiment_id"],
                "step_idx": row["step_idx"],
                "playout_idx": row["playout_idx"],
                "reward": row["reward"],
                "timestamp": row["timestamp"]
            }
            for row in cursor.fetchall()
        ]

    def list_experiments(
        self,
        search_method: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments, optionally filtered by search method.

        Args:
            search_method: Filter by "mcgs", "mcts", or "funsearch"
            limit: Maximum number of results

        Returns:
            List of experiment summaries (without full config)
        """
        if search_method:
            query = """SELECT id, name, search_method, started_at, completed_at,
                              final_reward, notes
                       FROM experiments
                       WHERE search_method = ?
                       ORDER BY started_at DESC"""
            params = (search_method,)
        else:
            query = """SELECT id, name, search_method, started_at, completed_at,
                              final_reward, notes
                       FROM experiments
                       ORDER BY started_at DESC"""
            params = ()

        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.execute(query, params)

        return [
            {
                "id": row["id"],
                "name": row["name"],
                "search_method": row["search_method"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "final_reward": row["final_reward"],
                "notes": row["notes"]
            }
            for row in cursor.fetchall()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.execute(
            """SELECT
                COUNT(*) as total_experiments,
                COUNT(CASE WHEN completed_at IS NOT NULL THEN 1 END) as completed,
                COUNT(CASE WHEN completed_at IS NULL THEN 1 END) as in_progress
               FROM experiments"""
        )
        exp_stats = cursor.fetchone()

        cursor = self.conn.execute("SELECT COUNT(*) as total_playouts FROM playouts")
        playout_stats = cursor.fetchone()

        cursor = self.conn.execute(
            """SELECT search_method, COUNT(*) as count
               FROM experiments
               GROUP BY search_method"""
        )
        by_method = {row["search_method"]: row["count"] for row in cursor.fetchall()}

        return {
            "total_experiments": exp_stats["total_experiments"],
            "completed_experiments": exp_stats["completed"],
            "in_progress_experiments": exp_stats["in_progress"],
            "total_playouts": playout_stats["total_playouts"],
            "experiments_by_method": by_method
        }

    def close(self):
        """Flush buffer and close database connection."""
        self._flush_playout_buffer()
        self.conn.close()
        self.logger.info("Database connection closed")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - ensures buffer is flushed."""
        self.close()


# Example usage
# python infrastructure/logging/experiment_db.py
if __name__ == "__main__":
    db = ExperimentDatabase("logs/experiments.db")
    experiments = db.list_experiments()
    print(experiments)
    db.close()
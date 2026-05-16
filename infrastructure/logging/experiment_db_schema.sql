-- Experiment tracking database schema
-- Minimal design for deterministic replay and analysis

-- Main experiments table: one row per experimental run
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                 -- Unique experiment name
    search_method TEXT NOT NULL,        -- "mcgs", "mcts"
    config JSON NOT NULL,               -- All parameters: hyperparameters, system_config, initial_algorithm, random_seed, etc.
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    final_algorithm JSON,               -- The best discovered algorithm
    final_reward REAL,                  -- Reward of the final algorithm
    notes TEXT
);

-- Playouts table: one row per search iteration
-- Contains only essential data for replay: experiment_id, step_idx, playout_idx, reward
CREATE TABLE IF NOT EXISTS playouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    step_idx INTEGER NOT NULL,          -- Which step in multi-step search (0 to new_steps-1)
    playout_idx INTEGER NOT NULL,       -- Which playout within this step (0 to n_playouts-1)
    reward REAL NOT NULL,               -- The reward obtained - key for deterministic replay
    timestamp TIMESTAMP NOT NULL,
    stat JSON,                          -- Flexible JSON dict for storing arbitrary stats and metadata
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);

-- Indexes for performance
-- Primary query pattern: get all playouts for an experiment/step in order
CREATE INDEX IF NOT EXISTS idx_playouts_experiment_step
    ON playouts(experiment_id, step_idx, playout_idx);

-- Secondary: get all playouts for an experiment across all steps
CREATE INDEX IF NOT EXISTS idx_playouts_experiment
    ON playouts(experiment_id);

-- For comparing search methods
CREATE INDEX IF NOT EXISTS idx_experiments_method
    ON experiments(search_method);

-- For temporal queries
CREATE INDEX IF NOT EXISTS idx_experiments_started
    ON experiments(started_at);

-- For finding experiments by name
CREATE INDEX IF NOT EXISTS idx_experiments_name
    ON experiments(name);

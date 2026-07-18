from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from a11.config import (
        AGENT_VERSION,
        BATCH_SIZE,
        CAPACITY_PER_DIM,
        CHECKPOINT_FILE,
        DATASET_DIR,
        DIMS_PER_UPDATE,
        ENV_COUNT,
        ENVS_PER_FILE,
        GOAL_THRESHOLD,
        LOG_EVERY,
        MAX_CONCURRENT_BACKEND_REDUCTIONS,
        NUM_GLOBALS,
        RESULTS_DIR,
        SAVE_EVERY,
        TOTAL_UPDATES,
        TRAIN_EVERY,
    )
    from a11.io_utils import gather_files, parse_dim_seed
    from a11.runtime import configure_main_runtime, get_device, seed_everything
else:
    from .config import (
        AGENT_VERSION,
        BATCH_SIZE,
        CAPACITY_PER_DIM,
        CHECKPOINT_FILE,
        DATASET_DIR,
        DIMS_PER_UPDATE,
        ENV_COUNT,
        ENVS_PER_FILE,
        GOAL_THRESHOLD,
        LOG_EVERY,
        MAX_CONCURRENT_BACKEND_REDUCTIONS,
        NUM_GLOBALS,
        RESULTS_DIR,
        SAVE_EVERY,
        TOTAL_UPDATES,
        TRAIN_EVERY,
    )
    from .io_utils import gather_files, parse_dim_seed
    from .runtime import configure_main_runtime, get_device, seed_everything


def main():
    configure_main_runtime()
    seed_everything()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    import torch

    if __package__ in (None, ""):
        from a11.agent import DQNAgent
        from a11.backend import LatticeBackend
        from a11.trainer import train_all
        from a11.workers import SubprocVecEnv
    else:
        from .agent import DQNAgent
        from .backend import LatticeBackend
        from .trainer import train_all
        from .workers import SubprocVecEnv

    LatticeBackend.validate_required_api()

    device = get_device()
    print(f"{AGENT_VERSION.upper()} learner device:", device)
    print("Visible learner GPUs:", torch.cuda.device_count())
    print("Backend:", LatticeBackend.module_info())
    print("Action = (pos, beta); external G6K runtime = disabled")
    print("ENV_COUNT=", ENV_COUNT, "| ENVS_PER_FILE=", ENVS_PER_FILE)
    print("MAX_CONCURRENT_BACKEND_REDUCTIONS=", MAX_CONCURRENT_BACKEND_REDUCTIONS)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    files = gather_files(DATASET_DIR)
    if not files:
        print("No dim/seed dataset .txt files found in", DATASET_DIR)
        raise SystemExit(1)

    composition = defaultdict(int)
    for filepath in files:
        composition[parse_dim_seed(filepath)[0]] += 1
    print(
        f"Dataset files: {len(files)} | all dimensions/seeds enabled | "
        f"composition={dict(sorted(composition.items()))}"
    )

    vec_env = SubprocVecEnv(
        files,
        env_count=ENV_COUNT,
        envs_per_file=ENVS_PER_FILE,
    )
    print(
        f"Active envs: {vec_env.num_envs} | "
        f"current dims: {sorted(set(vec_env.env_dims))}"
    )

    agent = DQNAgent(
        num_globals=NUM_GLOBALS,
        batch_size=BATCH_SIZE,
        dims_per_update=DIMS_PER_UPDATE,
        capacity_per_dim=CAPACITY_PER_DIM,
    )

    resume_path = os.path.join(RESULTS_DIR, CHECKPOINT_FILE)
    resume_extra = agent.load(resume_path)
    if resume_extra:
        print(f"Resumed {AGENT_VERSION.upper()} checkpoint:", resume_path)

    try:
        train_all(
            vec_env,
            agent,
            RESULTS_DIR,
            total_updates=TOTAL_UPDATES,
            train_every=TRAIN_EVERY,
            log_every=LOG_EVERY,
            save_every=SAVE_EVERY,
            goal_threshold=GOAL_THRESHOLD,
            resume_extra=resume_extra,
        )
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()

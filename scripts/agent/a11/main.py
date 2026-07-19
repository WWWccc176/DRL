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
        CPU_REDUCTION_CONCURRENCY,
        DATASET_DIR,
        DIM_MAX,
        DIM_MIN,
        DIMS_PER_UPDATE,
        ENV_COUNT,
        ENVS_PER_FILE,
        ENV_CPU_THREADS,
        GLOBAL_GPU_SIEVE_CONCURRENCY,
        GOAL_THRESHOLD,
        GPU_IDS,
        GPU_REDUCTIONS_PER_DEVICE,
        GPU_SIEVE_MIN_AVAILABLE_GB,
        LOG_EVERY,
        MAIN_CPU_THREADS,
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
        CPU_REDUCTION_CONCURRENCY,
        DATASET_DIR,
        DIM_MAX,
        DIM_MIN,
        DIMS_PER_UPDATE,
        ENV_COUNT,
        ENVS_PER_FILE,
        ENV_CPU_THREADS,
        GLOBAL_GPU_SIEVE_CONCURRENCY,
        GOAL_THRESHOLD,
        GPU_IDS,
        GPU_REDUCTIONS_PER_DEVICE,
        GPU_SIEVE_MIN_AVAILABLE_GB,
        LOG_EVERY,
        MAIN_CPU_THREADS,
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
    print("Action = (pos, beta); algorithm routing = native backend")
    print("ENV_COUNT=", ENV_COUNT, "| ENVS_PER_FILE=", ENVS_PER_FILE)
    print(
        "CPU scheduler:",
        f"concurrency={CPU_REDUCTION_CONCURRENCY}",
        f"env_threads={ENV_CPU_THREADS}",
        f"main_threads={MAIN_CPU_THREADS}",
    )
    print(
        "GPU scheduler:",
        f"physical_gpus={GPU_IDS}",
        f"global_heavy_sieves={GLOBAL_GPU_SIEVE_CONCURRENCY}",
        f"heavy_jobs_per_gpu={GPU_REDUCTIONS_PER_DEVICE}",
        f"mem_floor_gb={GPU_SIEVE_MIN_AVAILABLE_GB}",
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_files = gather_files(DATASET_DIR)
    files = [
        filepath
        for filepath in all_files
        if DIM_MIN <= parse_dim_seed(filepath)[0] <= DIM_MAX
    ]

    if not files:
        print("No dim/seed dataset .txt files found in", DATASET_DIR)
        raise SystemExit(1)

    composition = defaultdict(int)
    for filepath in files:
        composition[parse_dim_seed(filepath)[0]] += 1

    print("=" * 80)
    print("A11 DATASET CONFIGURATION")
    print("=" * 80)
    print(f"Dimension range : {DIM_MIN} - {DIM_MAX}")
    print(f"Dataset files   : {len(files)}")
    print(f"ENV_COUNT       : {ENV_COUNT}")
    print(f"ENVS_PER_FILE   : {ENVS_PER_FILE}")
    print(f"Dimensions      : {sorted(composition)}")
    print(f"Composition     : {dict(sorted(composition.items()))}")
    print("=" * 80, flush=True)

    vec_env = None

    try:
        vec_env = SubprocVecEnv(
            files,
            env_count=ENV_COUNT,
            envs_per_file=ENVS_PER_FILE,
        )

        print(
            f"Active envs: {vec_env.num_envs} | "
            f"current dims: {sorted(set(vec_env.env_dims))}"
        )
        print(
            "GPU env-slot assignment:",
            vec_env.gpu_assignment_counts,
        )

        agent = DQNAgent(
            num_globals=NUM_GLOBALS,
            batch_size=BATCH_SIZE,
            dims_per_update=DIMS_PER_UPDATE,
            capacity_per_dim=CAPACITY_PER_DIM,
        )

        resume_path = os.path.join(
            RESULTS_DIR,
            CHECKPOINT_FILE,
        )
        resume_extra = agent.load(resume_path)
        if resume_extra:
            print(
                f"Resumed {AGENT_VERSION.upper()} checkpoint:",
                resume_path,
            )

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

    except KeyboardInterrupt:
        # Normally trainer.py consumes Ctrl+C and saves the checkpoint first.
        # This catches interrupts during startup/reset before train_all owns control.
        print(
            "\n[A11] Ctrl+C received during startup. Closing workers.",
            flush=True,
        )

    finally:
        if vec_env is not None:
            vec_env.close()
        print("[A11] Worker shutdown complete.", flush=True)


if __name__ == "__main__":
    main()

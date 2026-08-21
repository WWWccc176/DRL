from __future__ import annotations

import json
import math
import os
import time

from .action_space import build_action_list
from .config import (
    CHECKPOINT_FILE,
    DETAIL_EVERY_CYCLES,
    ENVS_PER_FILE,
    ENUM_TIME_BUDGET_S,
    ENUM_TIMEOUT_GRACE_S,
    ENV_MAX_CONSECUTIVE_FAILURES,
    GPU_STEP_TIMEOUT_GRACE_S,
    SIEVE_MAX_ROUNDS,
    SIEVE_MIN_BETA,
    SIEVE_QUEUE_WAIT_TIMEOUT_S,
    SIEVE_RESPONSE_TIMEOUT_S,
    SIEVE_ONLY_ACTIONS,
    TRAIN_PROFILE,
)
from .results import (
    append_training_log,
    plot_training_history,
    save_dimension_summary,
    save_final_summary,
    save_seed_result,
)
from .runtime import log, status


def train_all(
    vec_env,
    agent,
    results_dir,
    total_updates=200000,
    train_every=4,
    log_every=4000,
    save_every=8000,
    goal_threshold=1.05,
    resume_extra=None,
):
    os.makedirs(results_dir, exist_ok=True)
    num_envs = vec_env.num_envs
    total_seeds = len(vec_env.dataset_pairs)
    dataset_dims = list(vec_env.dataset_dims)

    highest_dim = max(dataset_dims)
    highest_seed = min(
        seed for dim, seed in vec_env.dataset_pairs if dim == highest_dim
    )
    highest_action_space = build_action_list(highest_dim)
    highest_action_text = " -> ".join(
        f"({pos}, {beta})" for pos, beta in highest_action_space
    )

    global_best = {}
    global_info = {}
    history = {"loss": [], "best_min": []}
    updates = int((resume_extra or {}).get("updates", 0))
    env_steps = int((resume_extra or {}).get("env_steps", 0))
    cycles_completed = int((resume_extra or {}).get("cycles_completed", 0))

    if resume_extra:
        global_best.update(resume_extra.get("global_best", {}))
        global_info.update(resume_extra.get("global_info", {}))
        history = resume_extra.get("history", history)
        append_training_log(
            results_dir,
            "resume: replay buffer is intentionally not persisted; "
            "learner/network/optimizer/RNG state restored, replay starts empty",
        )

    latest_loss = float(history["loss"][-1]) if history["loss"] else 0.0
    interrupted = False
    pending: set[int] = set()

    progress_path = os.path.join(results_dir, "progress.json")
    progress_events_path = os.path.join(results_dir, "progress.log")

    restored_completed_runs = dict(
        (resume_extra or {}).get("completed_runs", {})
    )
    completed_runs = {
        str(key): int(value)
        for key, value in restored_completed_runs.items()
    }
    prior_cycles_unattributed = int(
        (resume_extra or {}).get(
            "prior_cycles_unattributed",
            cycles_completed if not restored_completed_runs else 0,
        )
    )
    active_progress: dict[int, dict] = {}

    def pair_key(dim: int, seed_id: int) -> str:
        return f"{int(dim)}:{int(seed_id)}"

    def max_steps_for_dim(dim: int) -> int:
        return math.ceil((int(dim) + 3) * int(dim) / 8)

    def next_run_index(dim: int, seed_id: int, exclude_env: int | None = None) -> int:
        key = pair_key(dim, seed_id)
        active_count = sum(
            1
            for env_id, entry in active_progress.items()
            if env_id != exclude_env
            and pair_key(entry["dim"], entry["seed_id"]) == key
        )
        return int(completed_runs.get(key, 0)) + active_count + 1

    def append_progress_event(event: str, **fields):
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            **fields,
        }
        with open(progress_events_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def write_progress_snapshot(reason: str):
        active_by_pair: dict[str, list[dict]] = {}
        for env_id, entry in active_progress.items():
            key = pair_key(entry["dim"], entry["seed_id"])
            active_by_pair.setdefault(key, []).append(
                {
                    "env_id": int(env_id),
                    "run_index": int(entry["run_index"]),
                    "step": int(entry["step"]),
                    "next_step": int(entry.get("next_step", entry["step"])),
                    "max_steps": int(entry["max_steps"]),
                    "state": str(entry["state"]),
                    "action_idx": entry.get("action_idx"),
                }
            )

        pairs = {}
        for dim, seed_id in vec_env.dataset_pairs:
            key = pair_key(dim, seed_id)
            pairs[key] = {
                "dim": int(dim),
                "seed_id": int(seed_id),
                "assigned_env_copies": int(ENVS_PER_FILE),
                "completed_episodes": int(completed_runs.get(key, 0)),
                "steps_per_episode": int(max_steps_for_dim(dim)),
                "sieve_rounds_per_sieve_action": int(SIEVE_MAX_ROUNDS),
                "active": sorted(
                    active_by_pair.get(key, []),
                    key=lambda item: item["env_id"],
                ),
            }

        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": reason,
            "updates": int(updates),
            "env_steps": int(env_steps),
            "cycles_completed": int(cycles_completed),
            "prior_cycles_unattributed": int(prior_cycles_unattributed),
            "dimension_range": [
                int(min(dataset_dims)),
                int(max(dataset_dims)),
            ],
            "dataset_pair_count": int(total_seeds),
            "assigned_env_copies_per_file": int(ENVS_PER_FILE),
            "pairs": pairs,
        }

        temporary = progress_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, progress_path)

    def checkpoint_extra():
        return {
            "updates": updates,
            "env_steps": env_steps,
            "cycles_completed": cycles_completed,
            "global_best": global_best,
            "global_info": global_info,
            "history": history,
            "completed_runs": completed_runs,
            "active_progress": active_progress,
            "prior_cycles_unattributed": prior_cycles_unattributed,
            "replay_persisted": False,
            "train_profile": TRAIN_PROFILE,
        }

    def save_checkpoint(reason: str):
        write_progress_snapshot(f"checkpoint:{reason}")
        path = os.path.join(results_dir, CHECKPOINT_FILE)
        agent.save(path, extra=checkpoint_extra())
        append_training_log(
            results_dir,
            f"checkpoint saved: reason={reason}, updates={updates}, env_steps={env_steps}",
        )

    def apply_best(best_update):
        key = (best_update["dim"], best_update["seed_id"])
        if best_update["ratio"] < global_best.get(key, float("inf")):
            first = key not in global_info
            global_best[key] = best_update["ratio"]
            global_info[key] = best_update

            infos = list(global_info.values())
            save_seed_result(results_dir, best_update, is_update=not first)
            save_dimension_summary(
                results_dir,
                best_update["dim"],
                infos,
                goal_threshold,
            )
            save_final_summary(results_dir, infos, goal_threshold)

            message = (
                f"★ dim{best_update['dim']} seed{best_update['seed_id']} "
                f"best={best_update['ratio']:.8f}"
            )
            log("  " + message)
            append_training_log(results_dir, message)

    def print_cycle_detail(cycle_no: int):
        dim_best = []
        for dim in dataset_dims:
            candidates = [info for info in global_info.values() if info["dim"] == dim]
            if candidates:
                best = min(candidates, key=lambda item: item["ratio"])
                dim_best.append(
                    f"dim{dim}: {best['ratio']:.8f} (seed={best['seed_id']})"
                )
            else:
                dim_best.append(f"dim{dim}: N/A")

        message = (
            f"\n===== cycle {cycle_no} =====\n"
            f"highest-dim first seed: dim={highest_dim}, seed={highest_seed}\n"
            f"full action space: {highest_action_text}\n"
            f"loss: {latest_loss:.8f}\n"
            f"min norm/GH by dimension: {' | '.join(dim_best)}\n"
        )
        print(message, flush=True)
        append_training_log(results_dir, message)

    def eps_now():
        # Exploration follows environment interaction, not whether a learner
        # update happened to return a positive loss. Scale by train_every so the
        # nominal decay horizon remains aligned with total_updates.
        decay_env_steps = max(1, int(total_updates) * max(1, int(train_every)))
        progress = min(1.0, env_steps / decay_env_steps)
        return max(0.05, 0.3 * (1.0 - progress))

    def action_timeout_seconds(entry: dict, action_idx: int) -> float:
        dim = int(entry["dim"])
        _, beta = build_action_list(dim)[int(action_idx)]
        gpu_timeout = (
            SIEVE_QUEUE_WAIT_TIMEOUT_S
            + SIEVE_RESPONSE_TIMEOUT_S
            + GPU_STEP_TIMEOUT_GRACE_S
        )
        if SIEVE_ONLY_ACTIONS or int(beta) >= int(SIEVE_MIN_BETA):
            timeout = gpu_timeout
        else:
            timeout = max(1.0, ENUM_TIME_BUDGET_S + ENUM_TIMEOUT_GRACE_S)

        # The terminal step performs the fixed GPU final polish after the agent
        # action, so its wall-clock envelope must include both operations.
        if int(entry.get("next_step", 0)) >= int(entry["max_steps"]):
            timeout += gpu_timeout
        return timeout

    try:
        states = vec_env.reset_all()
        state_by_eid = {env_id: states[env_id] for env_id in range(num_envs)}

        for env_id in range(num_envs):
            dim = int(vec_env.env_dims[env_id])
            seed_id = int(vec_env.env_seed_ids[env_id])
            key = pair_key(dim, seed_id)
            active_progress[env_id] = {
                "dim": dim,
                "seed_id": seed_id,
                "run_index": next_run_index(dim, seed_id, exclude_env=env_id),
                "step": 0,
                "next_step": 1,
                "max_steps": max_steps_for_dim(dim),
                "state": "ready",
                "action_idx": None,
            }

        write_progress_snapshot("startup")

        prev_s = [None] * num_envs
        prev_a = [None] * num_envs
        pending_since: dict[int, float] = {}
        pending_timeout: dict[int, float] = {}
        consecutive_failures = [0] * num_envs

        def dispatch_one(env_id: int, action_idx: int) -> None:
            vec_env.send_one(env_id, int(action_idx))
            pending.add(env_id)
            pending_since[env_id] = time.monotonic()
            pending_timeout[env_id] = action_timeout_seconds(
                active_progress[env_id],
                int(action_idx),
            )

        def recover_env(env_id: int, reason: str):
            pending.discard(env_id)
            pending_since.pop(env_id, None)
            pending_timeout.pop(env_id, None)
            consecutive_failures[env_id] += 1

            entry = active_progress[env_id]
            message = (
                f"env recovery: env={env_id}, dim={entry['dim']}, "
                f"seed={entry['seed_id']}, step={entry.get('next_step')}, "
                f"failure={consecutive_failures[env_id]}/"
                f"{ENV_MAX_CONSECUTIVE_FAILURES}, reason={reason}"
            )
            print("[A11] " + message, flush=True)
            append_training_log(results_dir, message)
            append_progress_event(
                "env_recovery",
                env_id=env_id,
                dim=entry["dim"],
                seed_id=entry["seed_id"],
                run_index=entry["run_index"],
                failed_step=entry.get("next_step"),
                action_idx=entry.get("action_idx"),
                consecutive_failures=consecutive_failures[env_id],
                reason=reason,
            )

            if consecutive_failures[env_id] > ENV_MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"env{env_id} exceeded "
                    f"A11_ENV_MAX_CONSECUTIVE_FAILURES="
                    f"{ENV_MAX_CONSECUTIVE_FAILURES}: {reason}"
                )

            state = vec_env.restart_one(env_id, filepath=vec_env.files[env_id])
            states[env_id] = state
            prev_s[env_id] = None
            prev_a[env_id] = None

            dim = int(vec_env.env_dims[env_id])
            seed_id = int(vec_env.env_seed_ids[env_id])
            active_progress[env_id] = {
                "dim": dim,
                "seed_id": seed_id,
                "run_index": entry["run_index"],
                "step": 0,
                "next_step": 1,
                "max_steps": max_steps_for_dim(dim),
                "state": "ready",
                "action_idx": None,
            }
            write_progress_snapshot("env_recovery")
            return state

        initial_actions = agent.act_envs(
            state_by_eid,
            eps_now(),
        )
        for env_id in range(num_envs):
            prev_s[env_id] = states[env_id]
            prev_a[env_id] = initial_actions[env_id]
            entry = active_progress[env_id]
            entry["state"] = "in_flight"
            entry["next_step"] = 1
            entry["action_idx"] = int(initial_actions[env_id])
            append_progress_event(
                "dispatch",
                env_id=env_id,
                dim=entry["dim"],
                seed_id=entry["seed_id"],
                run_index=entry["run_index"],
                step=entry["next_step"],
                max_steps=entry["max_steps"],
                assigned_env_copies=ENVS_PER_FILE,
                sieve_rounds=SIEVE_MAX_ROUNDS,
                action_idx=entry["action_idx"],
            )
            dispatch_one(env_id, initial_actions[env_id])

        t_start = time.time()

        while updates < total_updates:
            ready, dead = vec_env.poll_ready(list(pending))
            now = time.monotonic()
            timed_out = [
                env_id
                for env_id in pending
                if env_id in pending_since
                and now - pending_since[env_id]
                > pending_timeout.get(env_id, float("inf"))
            ]

            newly = {}
            detail_cycles = []
            successful_results = 0

            for env_id in sorted(set(dead) | set(timed_out)):
                if env_id in timed_out:
                    elapsed = now - pending_since.get(env_id, now)
                    reason = (
                        f"step wall-clock timeout after {elapsed:.1f}s "
                        f"(limit={pending_timeout.get(env_id, 0.0):.1f}s)"
                    )
                else:
                    process = vec_env.processes[env_id]
                    reason = f"worker exited with code {process.exitcode}"
                newly[env_id] = recover_env(env_id, reason)

            ready = [env_id for env_id in ready if env_id in pending]
            if not ready and not newly:
                time.sleep(0.0005)
                continue

            for env_id in ready:
                old_dim = int(vec_env.env_dims[env_id])
                old_seed_id = int(vec_env.env_seed_ids[env_id])
                old_key = pair_key(old_dim, old_seed_id)
                try:
                    obs, reward, done, info = vec_env.recv_one(env_id)
                except Exception as exc:
                    newly[env_id] = recover_env(
                        env_id,
                        f"receive failed: {type(exc).__name__}: {exc}",
                    )
                    continue
                pending.discard(env_id)
                pending_since.pop(env_id, None)
                pending_timeout.pop(env_id, None)
                consecutive_failures[env_id] = 0

                entry = active_progress[env_id]
                entry["step"] = int(info.get("step", entry["next_step"]))
                entry["next_step"] = entry["step"]
                entry["state"] = "episode_complete" if done else "ready"
                append_progress_event(
                    "result",
                    env_id=env_id,
                    dim=old_dim,
                    seed_id=old_seed_id,
                    run_index=entry["run_index"],
                    step=entry["step"],
                    max_steps=entry["max_steps"],
                    done=bool(done),
                    assigned_env_copies=ENVS_PER_FILE,
                    sieve_rounds=SIEVE_MAX_ROUNDS,
                    action_idx=entry.get("action_idx"),
                )

                best_update = info.pop("best_update", None)
                if best_update is not None:
                    apply_best(best_update)

                agent.remember(
                    old_dim,
                    prev_s[env_id],
                    prev_a[env_id],
                    reward,
                    obs,
                    done,
                )
                env_steps += 1
                successful_results += 1

                if done:
                    cycles_completed += 1
                    completed_runs[old_key] = int(completed_runs.get(old_key, 0)) + 1
                    append_progress_event(
                        "episode_complete",
                        env_id=env_id,
                        dim=old_dim,
                        seed_id=old_seed_id,
                        completed_episodes=completed_runs[old_key],
                        assigned_env_copies=ENVS_PER_FILE,
                        steps_per_episode=entry["max_steps"],
                        sieve_rounds=SIEVE_MAX_ROUNDS,
                    )
                    if cycles_completed % DETAIL_EVERY_CYCLES == 0:
                        detail_cycles.append(cycles_completed)

                    try:
                        next_state = vec_env.rotate_one(env_id)
                    except Exception as exc:
                        next_state = recover_env(
                            env_id,
                            f"file rotation failed: {type(exc).__name__}: {exc}",
                        )
                    states[env_id] = next_state
                    newly[env_id] = next_state

                    next_dim = int(vec_env.env_dims[env_id])
                    next_seed_id = int(vec_env.env_seed_ids[env_id])
                    active_progress[env_id] = {
                        "dim": next_dim,
                        "seed_id": next_seed_id,
                        "run_index": next_run_index(next_dim, next_seed_id, exclude_env=env_id),
                        "step": 0,
                        "next_step": 1,
                        "max_steps": max_steps_for_dim(next_dim),
                        "state": "ready",
                        "action_idx": None,
                    }
                    write_progress_snapshot("episode_complete")
                else:
                    states[env_id] = obs
                    newly[env_id] = obs

                if env_steps % train_every == 0:
                    loss = agent.learn()
                    if loss > 0:
                        latest_loss = float(loss)
                        updates += 1
                        history["loss"].append(latest_loss)
                        if updates % 500 == 0:
                            agent.step_scheduler()

            for cycle_no in detail_cycles:
                print_cycle_detail(cycle_no)

            actions = agent.act_envs(
                newly,
                eps_now(),
            )
            for env_id in newly:
                prev_s[env_id] = states[env_id]
                prev_a[env_id] = actions[env_id]

                entry = active_progress[env_id]
                entry["state"] = "in_flight"
                entry["next_step"] = min(
                    entry["step"] + 1,
                    entry["max_steps"],
                )
                entry["action_idx"] = int(actions[env_id])
                append_progress_event(
                    "dispatch",
                    env_id=env_id,
                    dim=entry["dim"],
                    seed_id=entry["seed_id"],
                    run_index=entry["run_index"],
                    step=entry["next_step"],
                    max_steps=entry["max_steps"],
                    assigned_env_copies=ENVS_PER_FILE,
                    sieve_rounds=SIEVE_MAX_ROUNDS,
                    action_idx=entry["action_idx"],
                )

                dispatch_one(env_id, actions[env_id])

            processed_count = successful_results
            if processed_count > 0 and env_steps % log_every < processed_count:
                best_min = min(global_best.values()) if global_best else float("inf")
                reached = sum(
                    1 for value in global_best.values() if value < goal_threshold
                )
                history["best_min"].append(best_min)
                rate = env_steps / max(
                    1e-6,
                    time.time() - t_start,
                )
                status(
                    f"upd {updates}/{total_updates} | "
                    f"cycles {cycles_completed} | "
                    f"ε{eps_now():.3f} | "
                    f"loss{latest_loss:.4f} | "
                    f"bestmin {best_min:.6f} | "
                    f"reached {reached}/{total_seeds} | "
                    f"{rate:.0f} env-steps/s"
                )
                write_progress_snapshot("periodic_status")

            if processed_count > 0 and env_steps % save_every < processed_count:
                save_checkpoint("periodic")

    except KeyboardInterrupt:
        interrupted = True
        message = (
            "\n[A11] Ctrl+C received by main process. "
            "Stop dispatching new actions, save checkpoint, then close workers."
        )
        print(message, flush=True)
        append_training_log(results_dir, message)

    finally:
        # Save learner/replay-related progress before main.py begins worker shutdown.
        # Do not wait for pending native reductions here; SubprocVecEnv.close() owns
        # the bounded close -> terminate -> kill sequence.
        try:
            save_checkpoint("interrupt" if interrupted else "final")
        except Exception as exc:
            print(
                f"[A11] checkpoint save failed during shutdown: {exc}",
                flush=True,
            )

    infos = list(global_info.values())
    for dim in dataset_dims:
        save_dimension_summary(
            results_dir,
            dim,
            infos,
            goal_threshold,
        )
    save_final_summary(
        results_dir,
        infos,
        goal_threshold,
    )

    if interrupted:
        print(
            "[A11] Training interrupted safely. Worker shutdown is handled by main.py.",
            flush=True,
        )
        return history

    # Normal completion only: briefly collect already-finished pending results.
    for env_id in list(pending):
        try:
            if vec_env.remotes[env_id].poll(2.0):
                _, _, _, info = vec_env.recv_one(env_id)
                best_update = info.pop("best_update", None)
                if best_update is not None:
                    apply_best(best_update)
        except Exception:
            pass

    plot_training_history(
        results_dir,
        history,
        goal_threshold,
    )

    print(
        "\nDone. Summary ->",
        os.path.join(results_dir, "summary.txt"),
    )
    return history

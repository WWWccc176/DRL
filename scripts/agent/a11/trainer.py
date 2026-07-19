from __future__ import annotations

import os
import time

from .action_space import build_action_list
from .config import AGENT_VERSION, DETAIL_EVERY_CYCLES
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

    latest_loss = float(history["loss"][-1]) if history["loss"] else 0.0
    interrupted = False
    pending: set[int] = set()

    def checkpoint_extra():
        return {
            "updates": updates,
            "env_steps": env_steps,
            "cycles_completed": cycles_completed,
            "global_best": global_best,
            "global_info": global_info,
            "history": history,
        }

    def save_checkpoint(reason: str):
        path = os.path.join(results_dir, f"{AGENT_VERSION}.pth")
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
        return max(
            0.05,
            0.3 * (1.0 - updates / max(1, total_updates)),
        )

    try:
        states = vec_env.reset_all()
        state_by_eid = {env_id: states[env_id] for env_id in range(num_envs)}
        prev_s = [None] * num_envs
        prev_a = [None] * num_envs

        initial_actions = agent.act_envs(
            state_by_eid,
            eps_now(),
        )
        for env_id in range(num_envs):
            prev_s[env_id] = states[env_id]
            prev_a[env_id] = initial_actions[env_id]
            vec_env.send_one(
                env_id,
                initial_actions[env_id],
            )
        pending = set(range(num_envs))

        t_start = time.time()

        while updates < total_updates:
            ready = vec_env.poll_ready(list(pending))
            if not ready:
                time.sleep(0.0005)
                continue

            newly = {}
            detail_cycles = []

            for env_id in ready:
                old_dim = vec_env.env_dims[env_id]
                obs, reward, done, info = vec_env.recv_one(env_id)
                pending.discard(env_id)

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

                if done:
                    cycles_completed += 1
                    if cycles_completed % DETAIL_EVERY_CYCLES == 0:
                        detail_cycles.append(cycles_completed)
                    next_state = vec_env.rotate_one(env_id)
                    states[env_id] = next_state
                    newly[env_id] = next_state
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
                vec_env.send_one(
                    env_id,
                    actions[env_id],
                )
                pending.add(env_id)

            if env_steps % log_every < len(ready):
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

            if env_steps % save_every < len(ready):
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

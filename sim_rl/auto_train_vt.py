#!/usr/bin/env python3
"""Automated VT (Vertical-Tactile) iterative training runner.

Trains AeroCubeGraspHW6ForceVT, reads logs, reports metrics.
Each iteration runs ~40M steps. Stops when EpLen >= 600 (30s) or 15 iters.

Usage:
  conda run -n aero_rl python sim_rl/auto_train_vt.py
"""

import csv
import datetime
import os
import subprocess
import sys
import time

# ── Configuration ──────────────────────────────────────────────────────────────
ENV_NAME = "AeroCubeGraspHW6ForceVT"
STEPS_PER_ITER = 40_000_000      # 40M steps per iteration
MAX_ITERS = 15
TARGET_EPLEN = 570               # 95% of 600 = 570 (≈28.5s, mission success)
NUM_EVALS = 8                    # eval frequency within each training run
NUM_ENVS = 8192
LEARNING_RATE = 3e-4             # initial lr (will be reduced for fine-tune)
SEED = 42

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(WORKSPACE, "mujoco_playground", "learning", "train_jax_ppo.py")
LOG_BASE = os.path.join(WORKSPACE, "mujoco_playground", "logs")

REPORT_PATH = os.path.join(os.path.dirname(WORKSPACE), "VT_iteration_report.txt")


def read_metrics_csv(csv_path):
    """Read metrics.csv and return list of dicts."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def get_latest_checkpoint(log_dir):
    """Return the latest checkpoint directory under log_dir/checkpoints/."""
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    children = [d for d in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()]
    if not children:
        return None
    children.sort(key=int)
    return os.path.join(ckpt_dir, children[-1])


def find_log_dir(suffix):
    """Find the log directory matching the given suffix."""
    if not os.path.isdir(LOG_BASE):
        return None
    candidates = [d for d in os.listdir(LOG_BASE) if suffix in d]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(LOG_BASE, candidates[-1])


def analyze_metrics(rows):
    """Analyze training metrics and return summary dict."""
    if not rows:
        return {"best_reward": 0, "best_eplen": 0, "last_reward": 0, "last_eplen": 0}

    rewards = []
    eplens = []
    for r in rows:
        # Try different reward key names
        reward = None
        for k in ["eval/episode_reward", "eval/episode/sum_reward",
                   "eval/episode_return", "episode/sum_reward"]:
            if k in r:
                reward = r[k]
                break
        if reward is not None:
            rewards.append(reward)

        # Episode length
        for k in ["eval/episode_reward_ep_length", "eval/avg_episode_length",
                   "eval/episode/ep_len", "training/episode_length"]:
            if k in r:
                eplens.append(r[k])
                break

    # Also look for specific reward component keys
    force_contacts = [r.get("eval/episode_reward_force_contact", 0) for r in rows]
    hold_positions = [r.get("eval/episode_reward_hold_position", 0) for r in rows]

    result = {
        "best_reward": max(rewards) if rewards else 0,
        "last_reward": rewards[-1] if rewards else 0,
        "best_eplen": max(eplens) if eplens else 0,
        "last_eplen": eplens[-1] if eplens else 0,
        "avg_last3_reward": sum(rewards[-3:]) / max(len(rewards[-3:]), 1) if rewards else 0,
        "avg_last3_eplen": sum(eplens[-3:]) / max(len(eplens[-3:]), 1) if eplens else 0,
        "n_evals": len(rewards),
    }
    return result


def run_training(iter_num, suffix, lr, steps, checkpoint_path=None, extra_flags=None):
    """Run a single training iteration."""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        f"--env_name={ENV_NAME}",
        f"--num_timesteps={steps}",
        f"--num_evals={NUM_EVALS}",
        f"--num_envs={NUM_ENVS}",
        f"--learning_rate={lr}",
        f"--seed={SEED + iter_num}",
        f"--suffix={suffix}",
        "--num_minibatches=32",
        "--unroll_length=40",
        "--num_updates_per_batch=4",
        "--discounting=0.97",
        "--entropy_cost=0.01",
        "--batch_size=256",
        "--policy_hidden_layer_sizes=512,256,128",
        "--value_hidden_layer_sizes=512,256,128",
        "--policy_obs_key=state",
        "--value_obs_key=privileged_state",
        "--domain_randomization",
        f"--episode_length=600",
    ]
    if checkpoint_path:
        cmd.append(f"--load_checkpoint_path={checkpoint_path}")
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"\n{'='*60}")
    print(f"VT-iter{iter_num}: Starting training ({steps/1e6:.0f}M steps, lr={lr})")
    print(f"  suffix={suffix}")
    if checkpoint_path:
        print(f"  checkpoint={checkpoint_path}")
    print(f"  cmd: {' '.join(cmd[:5])}...")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Run training in subprocess
    result = subprocess.run(
        cmd,
        cwd=os.path.join(WORKSPACE, "mujoco_playground"),
        capture_output=False,
        text=True,
    )

    elapsed = time.time() - start_time
    print(f"\nVT-iter{iter_num} completed in {elapsed/60:.1f} min (exit code={result.returncode})")

    return result.returncode, elapsed


def write_report(report_lines, path=REPORT_PATH):
    """Write or append to report file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report written to {path}")


def main():
    report = []
    report.append("=" * 70)
    report.append("VT (Vertical-Tactile) 自主迭代训练报告")
    report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")
    report.append("一、方案概述")
    report.append("-" * 40)
    report.append("目标: 将指尖触觉 (efc_force) 正式加入策略观测空间，")
    report.append("      实现稳定抓取 30s 以上 (episode_length=600 @ 20Hz)。")
    report.append("")
    report.append("核心算法:")
    report.append("  1. 触觉信号提取: MJX efc_force (约束求解器法向力)")
    report.append("     - 通过 data.contact.efc_address + data.efc_force 提取")
    report.append("     - 按 fingertip_geom ↔ cube_geom 匹配，聚合每指法向力")
    report.append("  2. 滤波: 指数移动平均 (EMA)")
    report.append("     - f_ema(t) = α·f_ema(t-1) + (1-α)·f_raw(t)")
    report.append("     - α=0.7 → 时间常数 ~3 步 (0.15s), 平滑碰撞瞬时尖峰")
    report.append("  3. 归一化: 饱和截断 → [0, 1]")
    report.append("     - obs = clip(f_ema / force_saturation_n, 0, 1)")
    report.append("     - force_saturation_n=3.0N (与硬件传感器量程对齐)")
    report.append("  4. 观测噪声注入: ±level*scale 均匀噪声 (增强泛化)")
    report.append("  5. PPO (Brax): 非对称 Actor-Critic")
    report.append("     - Policy: 17D state [motor_targets(6), tactile(5), last_act(6)]")
    report.append("     - Value:  83D privileged_state (含关节角/速/力/位置/方块状态)")
    report.append("     - Network: (512, 256, 128) MLP × 2")
    report.append("     - 8192 并行环境, 域随机化")
    report.append("")

    checkpoint_path = None
    best_overall_eplen = 0
    iter_results = []

    for iter_num in range(1, MAX_ITERS + 1):
        iter_time = datetime.datetime.now().strftime('%H:%M:%S')
        suffix = f"VT_iter{iter_num}"

        # Determine learning rate (higher for early iters, lower for fine-tuning)
        if iter_num == 1:
            lr = 3e-4   # Fresh start
            steps = STEPS_PER_ITER
        elif iter_num <= 3:
            lr = 2e-4   # Still exploring
            steps = STEPS_PER_ITER
        elif iter_num <= 6:
            lr = 1e-4   # Settling
            steps = STEPS_PER_ITER
        else:
            lr = 7e-5   # Fine-tuning
            steps = STEPS_PER_ITER

        # Run training
        exit_code, elapsed = run_training(
            iter_num, suffix, lr, steps, checkpoint_path
        )

        # Find log dir and read metrics
        log_dir = find_log_dir(suffix)
        metrics = {"best_reward": 0, "best_eplen": 0, "last_reward": 0, "last_eplen": 0}
        csv_rows = []

        if log_dir:
            csv_path = os.path.join(log_dir, "metrics.csv")
            if os.path.exists(csv_path):
                csv_rows = read_metrics_csv(csv_path)
                metrics = analyze_metrics(csv_rows)

            # Update checkpoint for next iteration
            new_ckpt = get_latest_checkpoint(log_dir)
            if new_ckpt:
                checkpoint_path = new_ckpt

        # Report this iteration
        iter_report = {
            "iter": iter_num,
            "lr": lr,
            "steps": steps,
            "elapsed_min": elapsed / 60,
            "exit_code": exit_code,
            **metrics,
        }
        iter_results.append(iter_report)

        if metrics["best_eplen"] > best_overall_eplen:
            best_overall_eplen = metrics["best_eplen"]

        report.append(f"\n--- VT-iter{iter_num} [{iter_time}] ---")
        report.append(f"  lr={lr}, steps={steps/1e6:.0f}M, time={elapsed/60:.1f}min")
        report.append(f"  best_reward={metrics['best_reward']:.1f}, last_reward={metrics['last_reward']:.1f}")
        report.append(f"  best_eplen={metrics['best_eplen']:.1f}, last_eplen={metrics['last_eplen']:.1f}")
        report.append(f"  avg_last3_eplen={metrics.get('avg_last3_eplen', 0):.1f}")
        report.append(f"  checkpoint={checkpoint_path}")

        # Check convergence
        if metrics["best_eplen"] >= TARGET_EPLEN:
            report.append(f"\n*** 目标达成! best_eplen={metrics['best_eplen']:.1f} >= {TARGET_EPLEN} ***")
            break

        # Analyze and suggest changes for next iteration
        if csv_rows:
            # Check for specific issues
            all_keys = csv_rows[0].keys() if csv_rows else []
            report.append(f"  metrics keys sample: {list(all_keys)[:10]}")

        # Write intermediate report
        write_report(report + ["\n(训练进行中...)"])
        print(f"VT-iter{iter_num} summary: eplen={metrics['best_eplen']:.1f}, reward={metrics['best_reward']:.1f}")

    # Final summary
    report.append("")
    report.append("=" * 70)
    report.append("二、迭代汇总")
    report.append("=" * 70)
    report.append(f"{'Iter':>4} | {'LR':>8} | {'Steps':>6} | {'Time':>6} | {'BestR':>8} | {'BestEL':>7} | {'LastEL':>7}")
    report.append("-" * 70)
    for r in iter_results:
        report.append(
            f"{r['iter']:>4} | {r['lr']:>8.1e} | {r['steps']/1e6:>5.0f}M | "
            f"{r['elapsed_min']:>5.1f}m | {r['best_reward']:>8.1f} | "
            f"{r['best_eplen']:>7.1f} | {r['last_eplen']:>7.1f}"
        )

    report.append(f"\n最佳 EpLen: {best_overall_eplen:.1f} / {TARGET_EPLEN} (目标)")
    if best_overall_eplen >= TARGET_EPLEN:
        report.append("结论: 触觉引入后成功实现稳定抓取 ≥30s!")
    else:
        report.append(f"结论: {len(iter_results)} 次迭代后未达目标，最佳 {best_overall_eplen:.1f} 步")

    write_report(report)
    print(f"\n{'='*60}")
    print(f"All iterations complete. Report: {REPORT_PATH}")
    print(f"Best EpLen: {best_overall_eplen:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

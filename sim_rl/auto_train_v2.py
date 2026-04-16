#!/usr/bin/env python3
"""Automated V2 灵犀手 iterative training runner.

Trains AeroCubeGraspV2Force with efc_force tactile, reads logs, reports metrics.
Each iteration runs ~40M steps. Stops when EpLen >= 480 (24s) or 15 iters.

Usage:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate aero_rl
  python sim_rl/auto_train_v2.py
"""

import csv
import datetime
import json
import os
import subprocess
import sys
import time
import argparse

# ── Configuration ──────────────────────────────────────────────────────────────
ENV_NAME = "AeroCubeGraspV2Force"
STEPS_PER_ITER = 40_000_000      # 40M steps per iteration
MAX_ITERS = 15
TARGET_EPLEN = 480               # 96% of 500 (≈24s @ 20Hz, mission success)
NUM_EVALS = 8                    # eval frequency within each training run
NUM_ENVS = 4096                  # V2 模型更轻量, 4096 足够
EPISODE_LENGTH = 500             # V2 环境 default
SEED = 42

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(WORKSPACE, "mujoco_playground", "learning", "train_jax_ppo.py")
LOG_BASE = os.path.join(WORKSPACE, "mujoco_playground", "logs")

REPORT_PATH = os.path.join(os.path.dirname(WORKSPACE), "V2_training_report.txt")
STRATEGY_LOG_PATH = os.path.join(os.path.dirname(WORKSPACE), "V2_training_analysis.txt")


def read_metrics_csv(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def get_latest_checkpoint(log_dir):
    """Return the checkpoints/ directory (not the numbered sub-dir).

    train_jax_ppo.py's _resolve_latest_checkpoint_dir() expects a parent
    directory and resolves the latest numbered sub-directory itself.
    """
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    children = [d for d in os.listdir(ckpt_dir)
                if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()]
    if not children:
        return None
    return ckpt_dir


def find_log_dir(suffix):
    if not os.path.isdir(LOG_BASE):
        return None
    candidates = [d for d in os.listdir(LOG_BASE) if suffix in d]
    if not candidates:
        return None
    candidates.sort()
    return os.path.join(LOG_BASE, candidates[-1])


def analyze_metrics(rows):
    if not rows:
        return {"best_reward": 0, "best_eplen": 0, "last_reward": 0, "last_eplen": 0}

    rewards = []
    eplens = []
    for r in rows:
        reward = None
        for k in ["eval/episode_reward", "eval/episode/sum_reward",
                   "eval/episode_return", "episode/sum_reward"]:
            if k in r:
                reward = r[k]
                break
        if reward is not None:
            rewards.append(reward)

        for k in ["eval/episode_reward_ep_length", "eval/avg_episode_length",
                   "eval/episode/ep_len", "training/episode_length"]:
            if k in r:
                eplens.append(r[k])
                break

    # Extract per-reward-component metrics
    force_contact = [r.get("eval/episode_reward_force_contact", 0) for r in rows]
    hold_position = [r.get("eval/episode_reward_hold_position", 0) for r in rows]
    stable_hold = [r.get("eval/episode_reward_stable_hold", 0) for r in rows]
    closure = [r.get("eval/episode_reward_closure", 0) for r in rows]

    result = {
        "best_reward": max(rewards) if rewards else 0,
        "last_reward": rewards[-1] if rewards else 0,
        "best_eplen": max(eplens) if eplens else 0,
        "last_eplen": eplens[-1] if eplens else 0,
        "avg_last3_reward": sum(rewards[-3:]) / max(len(rewards[-3:]), 1) if rewards else 0,
        "avg_last3_eplen": sum(eplens[-3:]) / max(len(eplens[-3:]), 1) if eplens else 0,
        "n_evals": len(rewards),
        "last_force_contact": force_contact[-1] if force_contact else 0,
        "last_hold_position": hold_position[-1] if hold_position else 0,
        "last_stable_hold": stable_hold[-1] if stable_hold else 0,
        "last_closure": closure[-1] if closure else 0,
    }
    return result


def run_training(iter_num, suffix, lr, steps, checkpoint_path=None, extra_flags=None):
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
        f"--episode_length={EPISODE_LENGTH}",
    ]
    if checkpoint_path:
        cmd.append(f"--load_checkpoint_path={checkpoint_path}")
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"\n{'='*60}")
    print(f"V2-iter{iter_num}: Starting training ({steps/1e6:.0f}M steps, lr={lr})")
    print(f"  suffix={suffix}")
    if checkpoint_path:
        print(f"  checkpoint={checkpoint_path}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=os.path.join(WORKSPACE, "mujoco_playground"),
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start_time
    print(f"\nV2-iter{iter_num} completed in {elapsed/60:.1f} min (exit code={result.returncode})")
    return result.returncode, elapsed


def write_report(report_lines, path=REPORT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Report written to {path}")


def write_strategy_log(analysis_lines, path=STRATEGY_LOG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_iter", type=int, default=1,
                        help="Iteration number to start from (skip previous)")
    parser.add_argument("--start_checkpoint", type=str, default=None,
                        help="Checkpoint directory to load for first iteration")
    args = parser.parse_args()

    report = []
    analysis = []
    report.append("=" * 70)
    report.append("V2 灵犀手方块抓握自主迭代训练报告")
    report.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")
    report.append("一、方案概述")
    report.append("-" * 40)
    report.append("目标: V2 灵犀手 (6通道直驱关节) 稳定抓握 3cm 方块 ≥24s")
    report.append("")
    report.append("核心组件:")
    report.append("  手: V2 灵犀手 (11关节, 6执行器, 直驱位置控制)")
    report.append("  触觉: efc_force (约束力提取, per-finger 聚合所有碰撞体)")
    report.append("  滤波: EMA α=0.7, 饱和截断 3.0N → [0,1]")
    report.append("  观测: state=17D [motor(6)+tactile(5)+last_act(6)]")
    report.append("        privileged=73D (含关节角/速/力/指尖位/方块状态)")
    report.append("  PPO: Brax, (512,256,128) MLP, 非对称 Actor-Critic")
    report.append(f"  并行环境: {NUM_ENVS}, 域随机化, episode_length={EPISODE_LENGTH}")
    report.append("")

    analysis.append("=" * 70)
    analysis.append("V2 训练策略分析日志")
    analysis.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analysis.append("=" * 70)

    checkpoint_path = args.start_checkpoint
    best_overall_eplen = 0
    iter_results = []

    for iter_num in range(args.start_iter, MAX_ITERS + 1):
        iter_time = datetime.datetime.now().strftime('%H:%M:%S')
        suffix = f"v2_R{iter_num:02d}"

        # Learning rate schedule
        if iter_num == 1:
            lr = 3e-4
        elif iter_num <= 3:
            lr = 2e-4
        elif iter_num <= 6:
            lr = 1e-4
        else:
            lr = 7e-5

        steps = STEPS_PER_ITER

        exit_code, elapsed = run_training(
            iter_num, suffix, lr, steps, checkpoint_path
        )

        log_dir = find_log_dir(suffix)
        metrics = {"best_reward": 0, "best_eplen": 0, "last_reward": 0, "last_eplen": 0}
        csv_rows = []

        if log_dir:
            csv_path = os.path.join(log_dir, "metrics.csv")
            if os.path.exists(csv_path):
                csv_rows = read_metrics_csv(csv_path)
                metrics = analyze_metrics(csv_rows)
            new_ckpt = get_latest_checkpoint(log_dir)
            if new_ckpt:
                checkpoint_path = new_ckpt

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

        # Report
        report.append(f"\n--- V2-iter{iter_num} [{iter_time}] ---")
        report.append(f"  lr={lr}, steps={steps/1e6:.0f}M, time={elapsed/60:.1f}min")
        report.append(f"  best_reward={metrics['best_reward']:.1f}, last_reward={metrics['last_reward']:.1f}")
        report.append(f"  best_eplen={metrics['best_eplen']:.1f}, last_eplen={metrics['last_eplen']:.1f}")
        report.append(f"  avg_last3_eplen={metrics.get('avg_last3_eplen', 0):.1f}")
        report.append(f"  force_contact={metrics.get('last_force_contact', 0):.3f}")
        report.append(f"  hold_position={metrics.get('last_hold_position', 0):.3f}")
        report.append(f"  stable_hold={metrics.get('last_stable_hold', 0):.3f}")
        report.append(f"  checkpoint={checkpoint_path}")

        # Strategy analysis
        analysis.append(f"\n--- V2-iter{iter_num} [{iter_time}] ---")
        analysis.append(f"  EpLen trend: best={metrics['best_eplen']:.1f}, last={metrics['last_eplen']:.1f}")
        analysis.append(f"  Reward components: force={metrics.get('last_force_contact',0):.3f}, "
                        f"hold={metrics.get('last_hold_position',0):.3f}, "
                        f"stable={metrics.get('last_stable_hold',0):.3f}, "
                        f"closure={metrics.get('last_closure',0):.3f}")

        # Convergence check
        if metrics["best_eplen"] >= TARGET_EPLEN:
            report.append(f"\n*** 目标达成! best_eplen={metrics['best_eplen']:.1f} >= {TARGET_EPLEN} ***")
            analysis.append(f"  >>> 目标达成!")
            break

        # Plateau detection (simple heuristic)
        if len(iter_results) >= 3:
            recent_eplens = [r["best_eplen"] for r in iter_results[-3:]]
            if max(recent_eplens) - min(recent_eplens) < 5 and max(recent_eplens) > 50:
                analysis.append(f"  [WARNING] EpLen 连续3轮无明显提升，可能进入平台期")

        # Write intermediate reports
        write_report(report + ["\n(训练进行中...)"])
        write_strategy_log(analysis)
        print(f"V2-iter{iter_num}: eplen={metrics['best_eplen']:.1f}, reward={metrics['best_reward']:.1f}")

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
        report.append("结论: V2 灵犀手成功实现稳定抓取 ≥24s!")
    else:
        report.append(f"结论: {len(iter_results)} 次迭代后未达目标，最佳 {best_overall_eplen:.1f} 步")

    write_report(report)
    write_strategy_log(analysis)
    print(f"\n{'='*60}")
    print(f"All iterations complete. Report: {REPORT_PATH}")
    print(f"Strategy log: {STRATEGY_LOG_PATH}")
    print(f"Best EpLen: {best_overall_eplen:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

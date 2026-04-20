#!/usr/bin/env python3
"""Quick metrics analyzer for VT training iterations."""
import csv
import sys
import os

def analyze(logdir):
    csv_path = os.path.join(logdir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"No metrics.csv in {logdir}")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No data rows found.")
        return

    print(f"{'Step':>12} {'EpLen':>7} {'Reward':>8} | {'hold_pos':>9} {'fc':>7} {'fo':>8} "
          f"{'stable':>7} {'contact':>8} {'grip':>6} {'surv':>6} | "
          f"{'term':>8} {'drop':>7} {'torq':>8} {'act_r':>7} | {'SPS':>6}")
    print("-" * 150)

    best_eplen = 0
    best_step = 0
    for r in rows:
        steps = int(r['num_steps'])
        eplen = float(r['eval/avg_episode_length'])
        reward = float(r['eval/episode_reward'])
        sps = float(r['eval/sps'])
        hold = float(r['eval/episode_reward/hold_position'])
        fc = float(r['eval/episode_reward/force_contact'])
        fo = float(r['eval/episode_reward/force_overload'])
        stable = float(r['eval/episode_reward/stable_hold'])
        contact = float(r['eval/episode_reward/contact'])
        grip = float(r['eval/episode_reward/grip_force'])
        surv = float(r['eval/episode_reward/survival'])
        term = float(r['eval/episode_reward/termination'])
        drop = float(r['eval/episode_reward/drop_risk'])
        torq = float(r['eval/episode_reward/torques'])
        act_r = float(r['eval/episode_reward/action_rate'])

        if eplen > best_eplen:
            best_eplen = eplen
            best_step = steps

        print(f"{steps:>12} {eplen:>7.1f} {reward:>8.1f} | {hold:>9.1f} {fc:>7.1f} {fo:>8.1f} "
              f"{stable:>7.1f} {contact:>8.1f} {grip:>6.1f} {surv:>6.1f} | "
              f"{term:>8.1f} {drop:>7.1f} {torq:>8.1f} {act_r:>7.1f} | {sps:>6.0f}")

    print(f"\nBest EpLen: {best_eplen:.1f} at step {best_step}")
    target = 570
    print(f"Target: {target} ({target*0.05:.1f}s)")
    if best_eplen >= target:
        print(">>> TARGET REACHED! <<<")
    else:
        print(f"Gap: {target - best_eplen:.1f} steps ({(target-best_eplen)*0.05:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find most recent log dir
        logbase = os.path.join(os.path.dirname(__file__), "..", "logs")
        dirs = sorted([d for d in os.listdir(logbase) if d.startswith("AeroCubeGraspHW6ForceVT")])
        if dirs:
            logdir = os.path.join(logbase, dirs[-1])
        else:
            print("No VT log directories found.")
            sys.exit(1)
    else:
        logdir = sys.argv[1]

    print(f"Analyzing: {logdir}\n")
    analyze(logdir)

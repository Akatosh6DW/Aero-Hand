"""Wait for training to finish, then dump full metrics."""
import subprocess, time, csv, glob, sys, os

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "v2_R07"

def is_running():
    """Check if train_jax_ppo with SUFFIX is running (exclude self)."""
    r = subprocess.run(
        ["pgrep", "-af", f"train_jax_ppo.*{SUFFIX}"],
        capture_output=True, text=True,
    )
    my_pid = str(os.getpid())
    for line in r.stdout.strip().split("\n"):
        if line.strip() and not line.startswith(my_pid):
            return True
    return False

print(f"[等待] 监控训练进程 '{SUFFIX}' ...")
poll = 0
while is_running():
    poll += 1
    if poll % 10 == 1:  # 每10分钟报告一次
        dirs = sorted(glob.glob(f"sim_rl/mujoco_playground/logs/*{SUFFIX}*/metrics.csv"))
        if dirs:
            with open(dirs[0]) as f:
                rows = list(csv.DictReader(f))
            if rows:
                r = rows[-1]
                step = int(float(r.get("num_steps", 0)))
                eplen = float(r.get("eval/avg_episode_length", 0))
                rew = float(r.get("eval/episode_reward", 0))
                print(f"  [{poll}] {step/1e6:.1f}M steps, EpLen={eplen:.1f}, R={rew:.1f}")
    time.sleep(60)

print(f"\n[完成] 训练进程 '{SUFFIX}' 已结束!")
print("=" * 70)

# Dump full metrics — use directory with most evals (skip crashed runs)
dirs = sorted(glob.glob(f"sim_rl/mujoco_playground/logs/*{SUFFIX}*/metrics.csv"))
if not dirs:
    print("ERROR: metrics.csv not found")
    sys.exit(1)

# Pick the directory with the most eval rows
best_csv = max(dirs, key=lambda d: sum(1 for _ in open(d)) - 1)

with open(best_csv) as f:
    rows = list(csv.DictReader(f))

print(f"共 {len(rows)} 个 eval 点\n")

# Key metrics table
header = f"{'Eval':>4} {'Steps':>8} {'EpLen':>7} {'Reward':>8} {'force_c':>8} {'stable_h':>9} {'finger_p':>9} {'force_b':>8} {'height':>7} {'hold_pos':>9}"
print(header)
print("-" * len(header))

for i, r in enumerate(rows):
    step = int(float(r.get("num_steps", 0)))
    eplen = float(r.get("eval/avg_episode_length", 0))
    rew = float(r.get("eval/episode_reward", 0))
    fc = float(r.get("training/reward/force_contact", 0))
    sh = float(r.get("training/reward/stable_hold", 0))
    fp = float(r.get("training/reward/finger_participation", 0))
    fb = float(r.get("training/reward/force_balance", 0))
    ht = float(r.get("training/reward/height", 0))
    hp = float(r.get("training/reward/hold_position", 0))
    print(f"{i+1:>4} {step/1e6:>7.1f}M {eplen:>7.1f} {rew:>8.1f} {fc:>8.1f} {sh:>9.1f} {fp:>9.1f} {fb:>8.1f} {ht:>7.1f} {hp:>9.1f}")

# Summary
ep_lens = [float(r.get("eval/avg_episode_length", 0)) for r in rows]
rewards = [float(r.get("eval/episode_reward", 0)) for r in rows]
print(f"\nEpLen: {ep_lens[0]:.1f} → {ep_lens[-1]:.1f} (max={max(ep_lens):.1f})")
print(f"Reward: {rewards[0]:.1f} → {rewards[-1]:.1f} (max={max(rewards):.1f})")
print(f"\n[DONE] 可以开始分析并设计下一轮迭代")

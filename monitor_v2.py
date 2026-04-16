#!/usr/bin/env python3
"""Quick V2 training monitor - reads latest metrics and prints summary."""
import csv, glob, sys, os

log_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sim_rl/mujoco_playground/logs")

suffix = sys.argv[1] if len(sys.argv) > 1 else "v2_R"
dirs = sorted(glob.glob(f'{log_base}/*{suffix}*'))

if not dirs:
    print(f"No log dirs matching '{suffix}'")
    sys.exit(1)

for d in dirs:
    name = os.path.basename(d).split('-')[-1]
    csv_path = f'{d}/metrics.csv'
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 10:
        print(f"--- {name}: NO DATA ---")
        continue
    
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    
    print(f"\n{'='*90}")
    print(f"--- {name} ({len(rows)} evals) ---")
    print(f"{'Steps':>10} {'EpLen':>6} {'Reward':>7} | {'force':>6} {'hold':>7} {'stable':>7} {'fp':>5} {'fb':>6} {'ht':>5} {'ct':>6} {'cl':>4} {'te':>5}")
    print("-"*90)
    for r in rows:
        s = int(float(r['num_steps']))
        el = float(r['eval/avg_episode_length'])
        rw = float(r['eval/episode_reward'])
        fc = float(r.get('eval/episode_reward/force_contact', 0))
        hp = float(r.get('eval/episode_reward/hold_position', 0))
        sh = float(r.get('eval/episode_reward/stable_hold', 0))
        fp = float(r.get('eval/episode_reward/finger_participation', 0))
        fb = float(r.get('eval/episode_reward/force_balance', 0))
        ht = float(r.get('eval/episode_reward/height', 0))
        ct = float(r.get('eval/episode_reward/contact', 0))
        cl = float(r.get('eval/episode_reward/closure', 0))
        te = float(r.get('eval/episode_reward/thumb_engage', 0))
        print(f'{s:>10} {el:>6.1f} {rw:>7.1f} | {fc:>6.1f} {hp:>7.1f} {sh:>7.1f} {fp:>5.1f} {fb:>6.1f} {ht:>5.1f} {ct:>6.1f} {cl:>4.1f} {te:>5.1f}')

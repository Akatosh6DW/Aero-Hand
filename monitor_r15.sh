#!/bin/bash
# Monitor R15 training progress
METRICS="/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2Force-20260413-022213-v2_R15/metrics.csv"
while kill -0 29412 2>/dev/null; do
  if [[ -f "$METRICS" ]]; then
    LINES=$(wc -l < "$METRICS")
    LAST=$(tail -1 "$METRICS" | cut -d',' -f1-3)
    echo "$(date +%H:%M:%S) lines=$LINES last=[$LAST]"
  fi
  sleep 300
done
echo "$(date +%H:%M:%S) R15 training finished!"
# Print final summary
python3 -c "
import csv
with open('$METRICS') as f:
    rows = list(csv.DictReader(f))
print(f'Total evals: {len(rows)}')
for r in rows:
    s = int(float(r['num_steps']))
    el = float(r['eval/avg_episode_length'])
    rw = float(r['eval/episode_reward'])
    print(f'  steps={s:>12,}  EpLen={el:.1f}  Reward={rw:.1f}')
best = max(rows, key=lambda r: float(r['eval/avg_episode_length']))
print(f'Best EpLen: {float(best[\"eval/avg_episode_length\"]):.1f} at step {int(float(best[\"num_steps\"])):,}')
"

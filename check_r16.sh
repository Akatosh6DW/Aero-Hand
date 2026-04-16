#!/bin/bash
R16_DIR="/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGraspV2Force-20260413-075957-v2_R16"
echo "=== R16 Status $(date) ==="
if [[ -f "$R16_DIR/metrics.csv" ]]; then
  lines=$(wc -l < "$R16_DIR/metrics.csv")
  echo "Metrics lines: $lines"
  if [[ $lines -gt 1 ]]; then
    echo "--- Progress ---"
    head -1 "$R16_DIR/metrics.csv" | cut -d',' -f1-3
    tail -3 "$R16_DIR/metrics.csv" | while IFS=',' read -r steps eplen reward rest; do
      printf "step=%-12s EpLen=%-10s Reward=%s\n" "$steps" "$eplen" "$reward"
    done
  fi
else
  echo "No metrics yet (still in JIT compilation)"
fi
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"

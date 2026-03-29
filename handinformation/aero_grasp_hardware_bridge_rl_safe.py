"""Aero bridge safe RL entrypoint.

用途：
- 启用 RL 推理，但以更保守的动作增益与限速启动。
- 启动前先开手到初态，减少与训练初始分布偏差。
"""

import sys

from aero_grasp_hardware_bridge import main as bridge_main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "rl"])
    if "--action_gain" not in sys.argv:
        sys.argv.extend(["--action_gain", "0.35"])
    if "--rl_preopen_s" not in sys.argv:
        sys.argv.extend(["--rl_preopen_s", "1.8"])
    if "--max_target_delta" not in sys.argv:
        sys.argv.extend(["--max_target_delta", "0.02"])
    if "--heartbeat_steps" not in sys.argv:
        sys.argv.extend(["--heartbeat_steps", "10"])
    bridge_main()

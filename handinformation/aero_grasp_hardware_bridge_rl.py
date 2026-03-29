"""Aero bridge RL entrypoint.

用途：
- 启用 RL 策略推理并通过串口下发 6 维控制。
- 用于真实策略部署联调。
"""

import sys

from aero_grasp_hardware_bridge import main as bridge_main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "rl"])
    bridge_main()

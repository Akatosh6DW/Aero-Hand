"""Aero bridge observe-only entrypoint.

用途：
- 仅执行串口反馈查询与CSV记录，不下发控制命令。
- 用于验证 0xF1 反馈对人工扰动是否可观测。
"""

import sys

from aero_grasp_hardware_bridge import main as bridge_main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "observe"])
    bridge_main()

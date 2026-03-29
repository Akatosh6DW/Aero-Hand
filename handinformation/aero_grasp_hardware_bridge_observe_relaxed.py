"""Aero bridge observe-relaxed entrypoint.

用途：
- 只执行反馈查询与CSV记录。
- 启动时尝试发送放松指令，降低位置保持影响。
- 用于验证手动掰动是否能进入 0xF1 可观测量。
"""

import sys

from aero_grasp_hardware_bridge import main as bridge_main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "observe_relaxed"])
    bridge_main()

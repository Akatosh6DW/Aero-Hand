"""Aero bridge debug entrypoint.

用途：
- 不跑 RL 推理，仅走硬件桥接链路并输出周期扰动。
- 用于快速确认串口、协议、映射、限速链路是否连续可动。
"""

import sys

from aero_grasp_hardware_bridge import main as bridge_main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "debug"])
    bridge_main()

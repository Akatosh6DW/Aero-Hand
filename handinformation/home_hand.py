"""归零脚本：将灵巧手所有关节缓慢恢复到全开（0°）位置。

用法：
  python handinformation/home_hand.py              # 默认 1s 缓动归零
  python handinformation/home_hand.py --ramp 2.0   # 2s 缓动
  python handinformation/home_hand.py --instant     # 直接发送 0°，不缓动
  python handinformation/home_hand.py --relax       # 仅发放松命令（使能=0），不控制角度
"""

import argparse
import time

import numpy as np
import serial

PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
DT = 0.02       # 控制周期 50Hz

FRAME_HEAD = 0x5A
FRAME_END  = 0x5D
CMD_MOTION = 0x10
FRAME_LEN  = 0x11


def _checksum(body: list[int]) -> int:
    return sum(body) & 0xFF


def _send(ser: serial.Serial, angles_deg: np.ndarray, enables: np.ndarray) -> None:
    """构造并发送一帧角度命令。"""
    d1_d2 = [CMD_MOTION, FRAME_LEN]
    payload: list[int] = []
    for i in range(6):
        payload.append(int(enables[i]))
        payload.append(int(np.clip(round(float(angles_deg[i])), 0, 90)))
    body = d1_d2 + payload
    pkt = bytes([FRAME_HEAD] + body + [_checksum(body), FRAME_END])
    ser.write(pkt)


def send_relax(ser: serial.Serial) -> None:
    """发使能=0放松帧。"""
    _send(ser, np.zeros(6), np.zeros(6, dtype=int))


def home(port: str, ramp_s: float, instant: bool, relax: bool) -> None:
    ser = serial.Serial(port, BAUDRATE, timeout=0.1)
    print(f"[home_hand] 串口已打开: {port} @ {BAUDRATE}")

    try:
        if relax:
            send_relax(ser)
            print("[home_hand] 已发送放松命令（使能=0）")
            return

        # 先读取当前下发角度——无法从硬件查询时直接从 90° 开始缓动
        # 保守起点：假设手当前在最大弯曲 90°，缓动回 0°
        start_deg = np.array([45.0, 90.0, 90.0, 90.0, 90.0, 90.0], dtype=np.float32)
        end_deg   = np.zeros(6, dtype=np.float32)
        enables   = np.ones(6, dtype=int)

        if instant:
            _send(ser, end_deg, enables)
            print("[home_hand] 已发送: 全部 0°（即时）")
        else:
            steps = max(1, round(ramp_s / DT))
            print(f"[home_hand] 缓动归零: {ramp_s}s / {steps} 步 → 0°")
            for k in range(steps + 1):
                t = k / steps
                deg = start_deg + t * (end_deg - start_deg)
                _send(ser, deg, enables)
                time.sleep(DT)
            print("[home_hand] 归零完成")

    finally:
        ser.close()
        print("[home_hand] 串口已关闭")


def main() -> None:
    p = argparse.ArgumentParser(description="灵巧手归零工具")
    p.add_argument("--port",    default=PORT)
    p.add_argument("--ramp",    type=float, default=1.5, help="缓动时间（秒），默认 1.5s")
    p.add_argument("--instant", action="store_true",    help="立即发送 0°，不缓动")
    p.add_argument("--relax",   action="store_true",    help="仅发放松命令（使能=0）")
    args = p.parse_args()
    home(args.port, args.ramp, args.instant, args.relax)


if __name__ == "__main__":
    main()

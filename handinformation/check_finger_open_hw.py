"""硬件版逐指张开测试脚本（0x5A 协议版）。

功能：握拳 → 拇指张开 → 食指 → 中指 → 无名指 → 小指 → 全开。
对标仿真脚本 check_finger_open_sequence.py，用于验证 ctrl↔硬件映射一致性。

协议：0x5A 帧，115200 波特率，6 通道 [使能, 字节值]。

用法：
  conda activate aero_rl
  python handinformation/check_finger_open_hw.py
  python handinformation/check_finger_open_hw.py --port /dev/ttyUSB1
  python handinformation/check_finger_open_hw.py --phase_s 2.5
"""

import argparse
import time

import numpy as np
import serial


# ──── 协议常量 ──── #
FRAME_HEAD = 0x5A
FRAME_END = 0x5D
CMD_MOTION = 0x10
FRAME_LEN_BYTE = 0x11  # 帧长度标识

# 6 通道名称（与训练 ctrl 一致）
CH_NAMES = ["thumb_rot", "thumb_flex", "index", "middle", "ring", "pinky"]

# MuJoCo ctrl 范围 (用于 ctrl→deg 转换)
CTRL_RANGES = np.array([
    [-0.1000, 1.7500],   # ch0 thumb_rot  (joint, radians)
    [ 0.0816, 0.1121],   # ch1 thumb_flex (tendon, meters)
    [ 0.0585, 0.1104],   # ch2 index
    [ 0.0585, 0.1104],   # ch3 middle
    [ 0.0585, 0.1104],   # ch4 ring
    [ 0.0585, 0.1104],   # ch5 pinky
], dtype=np.float64)


def deg_to_byte(angle_deg: float) -> int:
    """角度 0~90° → 协议字节 0~90（字节值=度数，协议V1.5）。"""
    a = float(np.clip(angle_deg, 0.0, 90.0))
    return int(round(a))


def build_motion_packet(deg6: np.ndarray, enables: np.ndarray = None) -> bytes:
    """构造 0x5A 动作帧 (17 字节)。"""
    if enables is None:
        enables = np.ones(6, dtype=np.int32)
    body = [CMD_MOTION, FRAME_LEN_BYTE]
    for i in range(6):
        body.append(int(enables[i]))
        body.append(deg_to_byte(float(deg6[i])))
    cksum = sum(body) & 0xFF
    return bytes([FRAME_HEAD] + body + [cksum, FRAME_END])


def send_deg6(ser: serial.Serial, deg6: np.ndarray) -> None:
    """发送 6 通道角度命令。"""
    ser.write(build_motion_packet(deg6))


def print_mapping(stage: str, deg6: np.ndarray) -> None:
    """打印 6 通道对照表。"""
    print(f"\n{'='*55}")
    print(f"  {stage}")
    print(f"{'='*55}")
    print(f"  {'通道':<14} {'角度(°)':>8} {'字节值':>7} {'比例%':>7}")
    print(f"  {'-'*14} {'-'*8} {'-'*7} {'-'*7}")
    for i in range(6):
        b = deg_to_byte(deg6[i])
        pct = b / 255.0 * 100.0
        print(f"  {CH_NAMES[i]:<14} {deg6[i]:>8.1f} {b:>7d} {pct:>6.1f}%")
    print()


def ramp_and_hold(
    ser: serial.Serial,
    from_deg: np.ndarray,
    to_deg: np.ndarray,
    ramp_s: float,
    hold_s: float,
    hz: float = 50.0,
) -> None:
    """平滑过渡 + 保持。"""
    steps = max(2, int(ramp_s * hz))
    dt = 1.0 / hz
    for s in range(steps):
        alpha = s / (steps - 1)
        cmd = from_deg + (to_deg - from_deg) * alpha
        send_deg6(ser, cmd)
        time.sleep(dt)
    send_deg6(ser, to_deg)
    time.sleep(hold_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="硬件版逐指张开测试 (0x5A协议)")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200,
                        help="波特率 (默认 115200)")
    parser.add_argument("--phase_s", type=float, default=1.8, help="每阶段过渡时间(秒)")
    parser.add_argument("--hold_s", type=float, default=0.8, help="每阶段保持时间(秒)")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baudrate, timeout=0.1)
    print(f"[hw_test] 串口已打开: {args.port} @ {args.baudrate}")
    print(f"[hw_test] 协议: 0x5A 帧, 6 通道, deg→byte(0-255)\n")

    # ── 握拳 / 全开 角度 ── #
    deg_fist = np.array([90.0, 90.0, 90.0, 90.0, 90.0, 90.0])
    deg_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print_mapping("握拳 (fist)", deg_fist)
    print_mapping("张开 (open)", deg_open)

    # 打印帧 hex 内容
    fist_pkt = build_motion_packet(deg_fist)
    open_pkt = build_motion_packet(deg_open)
    print(f"  fist hex: {fist_pkt.hex(' ')}")
    print(f"  open hex: {open_pkt.hex(' ')}\n")

    # ── 逐指序列 ── #
    # 6通道: [thumb_rot, thumb_flex, index, middle, ring, pinky]
    sequence = [
        ("拇指 (thumb_rot + thumb_flex)", [0, 1]),
        ("食指 (index)", [2]),
        ("中指 (middle)", [3]),
        ("无名指 (ring)", [4]),
        ("小指 (pinky)", [5]),
    ]

    try:
        print("[hw_test] 阶段 1/7: 握拳")
        current_deg = deg_fist.copy()
        send_deg6(ser, current_deg)
        time.sleep(args.phase_s)

        stage = 2
        for name, idxs in sequence:
            target_deg = current_deg.copy()
            for idx in idxs:
                target_deg[idx] = deg_open[idx]

            print(f"[hw_test] 阶段 {stage}/7: 张开 {name}")
            print_mapping(f"阶段 {stage}: 张开 {name}", target_deg)

            ramp_and_hold(ser, current_deg, target_deg, args.phase_s, args.hold_s)
            current_deg = target_deg.copy()
            stage += 1

        print("[hw_test] 阶段 7/7: 全部张开，保持")
        print_mapping("最终: 全开", current_deg)
        time.sleep(args.phase_s)

        print("[hw_test] 完成! 按 Ctrl+C 退出")
        while True:
            send_deg6(ser, current_deg)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[hw_test] 用户中断")
    finally:
        send_deg6(ser, deg_open)
        time.sleep(0.3)
        ser.close()
        print("[hw_test] 串口已关闭")


if __name__ == "__main__":
    main()

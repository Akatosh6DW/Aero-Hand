"""最简弯曲控制脚本（不引入力控、不接入RL）。

功能说明：
1. 通过串口发送 6 自由度“弯曲角度”命令。
2. 执行循环：张开 -> 弯曲 -> 张开。
3. 所有变量都用中文注释说明用途，便于后续对接和调参。

协议依据（串口/WIFI/Ethernet一致）：
- 帧头: 0x5A
- 指令: 0x10（运动控制）
- 帧长度: 0x11（固定17字节）
- 数据区: D3~D14，共12字节 = 6组 [使能, 角度]
- 校验: D1~D14 累加和低8位
- 帧尾: 0x5D
"""

import argparse
import time
from typing import Iterable

import numpy as np
import serial


class HandBendProtocol:
    """灵巧手基础弯曲协议封装。"""

    FRAME_HEAD = 0x5A
    CMD_MOTION = 0x10
    FRAME_LEN = 0x11
    FRAME_END = 0x5D

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.2):
        # 串口设备，例如 Linux 下常见 /dev/ttyUSB0
        self.port = port
        # 波特率，协议文档默认 115200
        self.baudrate = baudrate
        # 串口读取超时（本脚本几乎只发送，保留该参数方便后续扩展）
        self.timeout = timeout
        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)

    @staticmethod
    def _checksum(bytes_d1_to_d14: Iterable[int]) -> int:
        """校验和：D1~D14 累加和取低8位。"""
        return int(sum(bytes_d1_to_d14) & 0xFF)

    @staticmethod
    def _deg_to_u8(angle_deg: float) -> int:
        """将角度(0~90)映射到单字节整数。"""
        # 注意：协议定义角度0~90。若超出范围，这里会裁剪。
        a = float(np.clip(angle_deg, 0.0, 90.0))
        return int(round(a))

    def send_bend6(self, angles_deg: Iterable[float], enables: Iterable[int] | None = None) -> None:
        """发送6自由度弯曲命令。

        参数：
        - angles_deg: 6个角度，顺序为
          [拇指旋转, 拇指弯曲, 食指弯曲, 中指弯曲, 无名指弯曲, 小指弯曲]
        - enables: 6个使能位（0或1），默认全使能
        """
        angles = np.asarray(list(angles_deg), dtype=np.float32).reshape(6)

        if enables is None:
            en = np.ones(6, dtype=np.int32)
        else:
            en = np.asarray(list(enables), dtype=np.int32).reshape(6)
            en = np.where(en > 0, 1, 0)

        payload_d3_to_d14: list[int] = []
        for i in range(6):
            payload_d3_to_d14.append(int(en[i]))
            payload_d3_to_d14.append(self._deg_to_u8(float(angles[i])))

        d1_to_d14 = [self.CMD_MOTION, self.FRAME_LEN] + payload_d3_to_d14
        checksum = self._checksum(d1_to_d14)

        packet = [self.FRAME_HEAD] + d1_to_d14 + [checksum, self.FRAME_END]
        self.ser.write(bytes(packet))

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()


class SimpleBendController:
    """最基础弯曲控制器：只做开合，不做力控。"""

    def __init__(
        self,
        proto: HandBendProtocol,
        open_angle_deg: float = 0.0,
        close_angle_deg: float = 70.0,
        thumb_rotate_open_deg: float = 0.0,
        thumb_rotate_close_deg: float = 25.0,
        ramp_time_sec: float = 0.8,
        hold_open_sec: float = 0.6,
        hold_close_sec: float = 1.0,
        control_hz: float = 30.0,
    ):
        self.proto = proto

        # open_angle_deg: 非拇指旋转通道在“张开”时的弯曲角度。
        self.open_angle_deg = float(open_angle_deg)
        # close_angle_deg: 非拇指旋转通道在“闭合”时的弯曲角度。
        self.close_angle_deg = float(close_angle_deg)

        # thumb_rotate_open_deg: 拇指旋转通道张开角。
        self.thumb_rotate_open_deg = float(thumb_rotate_open_deg)
        # thumb_rotate_close_deg: 拇指旋转通道闭合角（对捏常需要拇指内收）。
        self.thumb_rotate_close_deg = float(thumb_rotate_close_deg)

        # ramp_time_sec: 张开<->闭合过渡时间，越大越平滑。
        self.ramp_time_sec = float(ramp_time_sec)
        # hold_open_sec: 张开姿态保持时间。
        self.hold_open_sec = float(hold_open_sec)
        # hold_close_sec: 闭合姿态保持时间。
        self.hold_close_sec = float(hold_close_sec)

        # control_hz: 过渡阶段发送频率。
        self.control_hz = float(control_hz)

    def _open_pose(self) -> np.ndarray:
        # 顺序: [拇指旋转, 拇指弯曲, 食指, 中指, 无名指, 小指]
        return np.array([
            self.thumb_rotate_open_deg,
            self.open_angle_deg,
            self.open_angle_deg,
            self.open_angle_deg,
            self.open_angle_deg,
            self.open_angle_deg,
        ], dtype=np.float32)

    def _close_pose(self) -> np.ndarray:
        return np.array([
            self.thumb_rotate_close_deg,
            self.close_angle_deg,
            self.close_angle_deg,
            self.close_angle_deg,
            self.close_angle_deg,
            self.close_angle_deg,
        ], dtype=np.float32)

    def _move_linear(self, start_pose: np.ndarray, end_pose: np.ndarray) -> None:
        # 线性插值，让动作更平滑，减少机械冲击。
        steps = max(2, int(self.ramp_time_sec * self.control_hz))
        dt = 1.0 / self.control_hz
        for i in range(steps):
            alpha = i / (steps - 1)
            cmd = (1.0 - alpha) * start_pose + alpha * end_pose
            self.proto.send_bend6(cmd)
            time.sleep(dt)

    def run_cycles(self, cycles: int) -> None:
        p_open = self._open_pose()
        p_close = self._close_pose()

        # 启动先回到张开位，防止上电状态未知。
        self.proto.send_bend6(p_open)
        time.sleep(0.3)

        for _ in range(int(cycles)):
            # 张开保持
            self.proto.send_bend6(p_open)
            time.sleep(self.hold_open_sec)

            # 张开 -> 闭合
            self._move_linear(p_open, p_close)
            time.sleep(self.hold_close_sec)

            # 闭合 -> 张开
            self._move_linear(p_close, p_open)

        # 结束时回到张开位。
        self.proto.send_bend6(p_open)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最简单弯曲控制（不引入力控）")

    # 串口参数
    parser.add_argument("--port", default="/dev/ttyUSB0", help="串口设备，例如 /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200, help="串口波特率")

    # 角度参数（单位：度）
    parser.add_argument("--open_angle_deg", type=float, default=0.0, help="四指+拇指弯曲通道张开角")
    parser.add_argument("--close_angle_deg", type=float, default=70.0, help="四指+拇指弯曲通道闭合角")
    parser.add_argument("--thumb_rotate_open_deg", type=float, default=0.0, help="拇指旋转通道张开角")
    parser.add_argument("--thumb_rotate_close_deg", type=float, default=25.0, help="拇指旋转通道闭合角")

    # 时序参数
    parser.add_argument("--ramp_time_sec", type=float, default=0.8, help="开合过渡时间")
    parser.add_argument("--hold_open_sec", type=float, default=0.6, help="张开保持时间")
    parser.add_argument("--hold_close_sec", type=float, default=1.0, help="闭合保持时间")
    parser.add_argument("--control_hz", type=float, default=30.0, help="过渡阶段发送频率")

    # 循环次数
    parser.add_argument("--cycles", type=int, default=10, help="开合循环次数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    proto = HandBendProtocol(
        port=args.port,
        baudrate=args.baudrate,
    )

    controller = SimpleBendController(
        proto=proto,
        open_angle_deg=args.open_angle_deg,
        close_angle_deg=args.close_angle_deg,
        thumb_rotate_open_deg=args.thumb_rotate_open_deg,
        thumb_rotate_close_deg=args.thumb_rotate_close_deg,
        ramp_time_sec=args.ramp_time_sec,
        hold_open_sec=args.hold_open_sec,
        hold_close_sec=args.hold_close_sec,
        control_hz=args.control_hz,
    )

    try:
        controller.run_cycles(args.cycles)
    finally:
        proto.close()


if __name__ == "__main__":
    main()

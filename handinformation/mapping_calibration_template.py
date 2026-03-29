"""7维策略到6维硬件映射的标定采集模板。

用途：
1. 逐通道扫描硬件6自由度命令（0~90度）。
2. 通过 0xF1 查询回读角度，记录命令值与回读值。
3. 导出 CSV，供后续拟合映射（线性/LUT/矩阵）使用。

说明：
- 本模板不接入 RL，仅做“硬件通道标定数据采集”。
- 如果你的设备应答节拍不同，请按现场协议调 read_response() 超时与帧头搜索逻辑。
"""

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import serial


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 0.2


class HandProtocolV15:
    """基于 V1.5 串口协议的最小实现（动作控制 + 角度查询）。"""

    HEAD = 0x5A
    TAIL = 0x5D

    CMD_MOTION = 0x10
    CMD_QUERY_ANGLE = 0xF1

    LEN_MOTION = 0x11  # 17字节帧
    LEN_QUERY = 0x07

    def __init__(self, cfg: SerialConfig):
        self.ser = serial.Serial(cfg.port, cfg.baudrate, timeout=cfg.timeout)

    @staticmethod
    def checksum(d1_to_dn: Iterable[int]) -> int:
        return int(sum(d1_to_dn) & 0xFF)

    def send_motion_angles(self, angles6_deg: Iterable[float], enables6: Optional[Iterable[int]] = None) -> None:
        """发送动作控制。

        6自由度顺序：
        [拇指旋转, 拇指弯曲, 食指弯曲, 中指弯曲, 无名指弯曲, 小指弯曲]
        """
        angles = np.asarray(list(angles6_deg), dtype=np.float32).reshape(6)
        angles = np.clip(angles, 0.0, 90.0).astype(np.int32)

        if enables6 is None:
            enables = np.ones(6, dtype=np.int32)
        else:
            enables = np.asarray(list(enables6), dtype=np.int32).reshape(6)
            enables = np.where(enables > 0, 1, 0)

        payload = []
        for i in range(6):
            payload.extend([int(enables[i]), int(angles[i])])

        d1_to_d14 = [self.CMD_MOTION, self.LEN_MOTION] + payload
        d15 = self.checksum(d1_to_d14)
        frame = [self.HEAD] + d1_to_d14 + [d15, self.TAIL]
        self.ser.write(bytes(frame))

    def send_query_angle(self) -> None:
        # 查询帧格式：5A F1 07 00 00 CHK 5D（按你的总结）
        d1_to_d5 = [self.CMD_QUERY_ANGLE, self.LEN_QUERY, 0x00, 0x00]
        chk = self.checksum(d1_to_d5)
        frame = [self.HEAD] + d1_to_d5 + [chk, self.TAIL]
        self.ser.write(bytes(frame))

    def read_response(self, expected_len: int, timeout_s: float = 0.25) -> Optional[bytes]:
        """读取指定长度的一帧。简单实现，便于你按现场协议细化。"""
        t0 = time.time()
        buf = bytearray()
        while time.time() - t0 < timeout_s:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)
                # 找帧头
                while len(buf) >= expected_len:
                    if buf[0] != self.HEAD:
                        buf.pop(0)
                        continue
                    candidate = bytes(buf[:expected_len])
                    if candidate[-1] == self.TAIL:
                        return candidate
                    buf.pop(0)
        return None

    @staticmethod
    def parse_angle_reply(frame12: bytes) -> Optional[np.ndarray]:
        """解析 0xF1 应答 12 字节，返回6路角度。

        预期结构：
        D0=5A, D1=F1, D2=0C, D3=00, D4~D9=6路角度, D10=CHK, D11=5D
        """
        if len(frame12) != 12:
            return None
        if frame12[0] != 0x5A or frame12[11] != 0x5D:
            return None
        d1_to_d9 = list(frame12[1:10])
        chk = int(frame12[10])
        if (sum(d1_to_d9) & 0xFF) != chk:
            return None
        # 角度回读区是6个原始字节，先按 uint8 解包再转浮点。
        return np.frombuffer(bytes(frame12[4:10]), dtype=np.uint8).astype(np.float32)

    def query_angles_once(self) -> Optional[np.ndarray]:
        self.send_query_angle()
        frame = self.read_response(expected_len=12)
        if frame is None:
            return None
        return self.parse_angle_reply(frame)

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()


def sweep_one_channel(
    proto: HandProtocolV15,
    ch_idx: int,
    cmd_values: np.ndarray,
    hold_s: float,
    repeat_query: int,
    base_angles6: np.ndarray,
    switch_settle_s: float,
    switch_discard_query: int,
    query_interval_s: float,
    aggregate: str,
    cycle_idx: int,
    direction: str,
):
    """单通道扫描并回读。"""
    rows = []

    # 切换通道时先回到基准位并等待稳定，降低上一通道残留对首点的污染。
    proto.send_motion_angles(base_angles6)
    time.sleep(max(0.0, float(switch_settle_s)))
    for _ in range(max(0, int(switch_discard_query))):
        _ = proto.query_angles_once()
        time.sleep(max(0.0, float(query_interval_s)))

    for cmd in cmd_values:
        cmd_pose = base_angles6.copy()
        cmd_pose[ch_idx] = float(cmd)

        proto.send_motion_angles(cmd_pose)
        time.sleep(hold_s)

        # 多次查询取均值，减小抖动影响
        rb_list = []
        for _ in range(repeat_query):
            rb = proto.query_angles_once()
            if rb is not None:
                rb_list.append(rb)
            time.sleep(max(0.0, float(query_interval_s)))

        if rb_list:
            stack = np.stack(rb_list, axis=0)
            if aggregate == "median":
                rb_mean = np.median(stack, axis=0)
            else:
                rb_mean = np.mean(stack, axis=0)
        else:
            rb_mean = np.full(6, np.nan, dtype=np.float32)

        rows.append({
            "timestamp": time.time(),
            "cycle": cycle_idx,
            "direction": direction,
            "sweep_channel": ch_idx,
            "cmd_thumb_rot": cmd_pose[0],
            "cmd_thumb_flex": cmd_pose[1],
            "cmd_index": cmd_pose[2],
            "cmd_middle": cmd_pose[3],
            "cmd_ring": cmd_pose[4],
            "cmd_pinky": cmd_pose[5],
            "rb_thumb_rot": rb_mean[0],
            "rb_thumb_flex": rb_mean[1],
            "rb_index": rb_mean[2],
            "rb_middle": rb_mean[3],
            "rb_ring": rb_mean[4],
            "rb_pinky": rb_mean[5],
        })

    return rows


def save_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "cycle",
        "direction",
        "sweep_channel",
        "cmd_thumb_rot", "cmd_thumb_flex", "cmd_index", "cmd_middle", "cmd_ring", "cmd_pinky",
        "rb_thumb_rot", "rb_thumb_flex", "rb_index", "rb_middle", "rb_ring", "rb_pinky",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args():
    p = argparse.ArgumentParser(description="硬件6通道标定采集模板（支持VS Code一键运行）")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--out_csv", default="/home/ll/SRTP/Aero-Hand/handinformation/calib_hw6_hiacc.csv")
    p.add_argument("--min_deg", type=float, default=0.0)
    p.add_argument("--max_deg", type=float, default=90.0)
    p.add_argument("--step_deg", type=float, default=15.0)
    p.add_argument("--hold_s", type=float, default=0.3)
    p.add_argument("--repeat_query", type=int, default=5)
    p.add_argument("--base_deg", type=float, default=0.0, help="非扫描通道保持角度")
    p.add_argument("--switch_settle_s", type=float, default=0.5, help="切换扫描通道后的稳定等待时间")
    p.add_argument("--switch_discard_query", type=int, default=5, help="切换扫描通道后丢弃的查询帧数")
    p.add_argument("--include_reverse", action="store_true", help="每轮在正向扫描后追加反向扫描")
    p.add_argument("--no_reverse", action="store_false", dest="include_reverse", help="关闭反向扫描")
    p.add_argument("--cycles", type=int, default=3, help="每个通道重复轮数")
    p.add_argument("--query_interval_s", type=float, default=0.015, help="连续查询之间的间隔")
    p.add_argument("--aggregate", choices=["mean", "median"], default="median", help="多次回读聚合方式")
    p.set_defaults(include_reverse=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SerialConfig(port=args.port, baudrate=args.baudrate)
    proto = HandProtocolV15(cfg)

    cmd_values = np.arange(args.min_deg, args.max_deg + 1e-6, args.step_deg, dtype=np.float32)
    base_angles6 = np.full(6, float(args.base_deg), dtype=np.float32)

    all_rows = []
    try:
        for ch in range(6):
            for cycle_idx in range(max(1, int(args.cycles))):
                rows = sweep_one_channel(
                    proto=proto,
                    ch_idx=ch,
                    cmd_values=cmd_values,
                    hold_s=args.hold_s,
                    repeat_query=args.repeat_query,
                    base_angles6=base_angles6,
                    switch_settle_s=args.switch_settle_s,
                    switch_discard_query=args.switch_discard_query,
                    query_interval_s=args.query_interval_s,
                    aggregate=args.aggregate,
                    cycle_idx=cycle_idx,
                    direction="forward",
                )
                all_rows.extend(rows)

                if args.include_reverse:
                    rows = sweep_one_channel(
                        proto=proto,
                        ch_idx=ch,
                        cmd_values=cmd_values[::-1],
                        hold_s=args.hold_s,
                        repeat_query=args.repeat_query,
                        base_angles6=base_angles6,
                        switch_settle_s=args.switch_settle_s,
                        switch_discard_query=args.switch_discard_query,
                        query_interval_s=args.query_interval_s,
                        aggregate=args.aggregate,
                        cycle_idx=cycle_idx,
                        direction="reverse",
                    )
                    all_rows.extend(rows)

        save_csv(all_rows, Path(args.out_csv))
        print(f"Saved calibration rows: {len(all_rows)} -> {args.out_csv}")
    finally:
        proto.close()


if __name__ == "__main__":
    main()

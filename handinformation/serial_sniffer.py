"""串口代理嗅探工具 — 拦截官方 GUI 与灵巧手之间的通信。

原理：
  真实串口 <──> 本脚本(代理) <──> 虚拟串口pty <──> 官方GUI

使用方法：
  1. 运行本脚本：
       python handinformation/serial_sniffer.py --real /dev/ttyUSB0
     脚本会打印虚拟串口路径，例如 /dev/pts/3

  2. 打开官方 GUI（sdk/src/aero_open_sdk/gui.py），
     端口选择上面打印的虚拟路径（如 /dev/pts/3），波特率 921600。

  3. 在 GUI 中操作（拖滑条到握拳位置等），
     本脚本会实时打印所有 TX/RX 帧的十六进制内容和解析。

  4. Ctrl+C 退出。

也可以 --dump 模式：不创建代理，直接监听串口上的数据：
  python handinformation/serial_sniffer.py --real /dev/ttyUSB0 --dump
"""

import argparse
import os
import select
import struct
import sys
import time

import serial

# ─── 帧解析 ─── #
OPCODES = {
    0x01: "HOMING",
    0x02: "SET_ID",
    0x03: "TRIM",
    0x11: "CTRL_POS",
    0x12: "CTRL_TOR",
    0x21: "GET_ALL",
    0x22: "GET_POS",
    0x23: "GET_VEL",
    0x24: "GET_CURR",
    0x25: "GET_TEMP",
    0x31: "SET_SPE",
    0x32: "SET_TOR",
}

SLIDER_LABELS = [
    "thumb_abd", "thumb_flex", "thumb_tendon",
    "index", "middle", "ring", "pinky",
]


def parse_frame(data: bytes, direction: str = "TX") -> str:
    """解析 16 字节帧，返回人类可读字符串。"""
    if len(data) != 16:
        return f"[{direction}] raw({len(data)}B): {data.hex(' ')}"

    opcode = data[0]
    name = OPCODES.get(opcode, f"UNK_0x{opcode:02X}")
    payload = data[2:]  # skip filler byte

    parts = [f"[{direction}] {name}(0x{opcode:02X})"]

    if opcode == 0x11:  # CTRL_POS
        values = struct.unpack_from("<7H", payload)
        for i, v in enumerate(values):
            pct = v / 65535.0 * 100.0
            label = SLIDER_LABELS[i] if i < len(SLIDER_LABELS) else f"ch{i}"
            parts.append(f"  {label}: {v:5d} ({pct:5.1f}%)")
    elif opcode == 0x22:  # GET_POS response
        values = struct.unpack_from("<7H", payload)
        for i, v in enumerate(values):
            label = SLIDER_LABELS[i] if i < len(SLIDER_LABELS) else f"ch{i}"
            parts.append(f"  {label}: {v:5d}")
    else:
        parts.append(f"  payload: {payload.hex(' ')}")

    parts.append(f"  hex: {data.hex(' ')}")
    return "\n".join(parts)


def print_our_fist_comparison():
    """打印我们旧脚本发送的字节 vs SDK 正确字节的对比。"""
    print("\n" + "=" * 70)
    print("  对比：旧协议(我们的脚本) vs 新协议(SDK/固件)")
    print("=" * 70)

    # 旧协议：握拳 = 全90° → deg_to_u8(90) = round(90/90*255) = 255
    old_body = [0x10, 0x11]
    for _ in range(6):
        old_body.extend([0x01, 0xFF])  # enable=1, angle=255
    cksum = sum(old_body) & 0xFF
    old_packet = bytes([0x5A] + old_body + [cksum, 0x5D])

    print(f"\n旧协议 (17 bytes, 115200 baud):")
    print(f"  hex: {old_packet.hex(' ')}")
    print(f"  帧头=0x5A, cmd=0x10, len=0x11, 6×[en=0x01,angle=0xFF], cksum, 帧尾=0x5D")

    # 新协议：握拳 = 7×65535
    new_packet = struct.pack("<2B7H", 0x11, 0x00,
                             65535, 65535, 65535, 65535, 65535, 65535, 65535)
    print(f"\n新协议 CTRL_POS (16 bytes, 921600 baud):")
    print(f"  hex: {new_packet.hex(' ')}")
    print(f"  opcode=0x11, filler=0x00, 7×uint16=65535(0xFFFF)")

    print(f"\n关键差异:")
    print(f"  1. 帧长度: 17 vs 16 字节 → 固件 16B 对齐被破坏")
    print(f"  2. 波特率: 115200 vs 921600 → 数据乱码")
    print(f"  3. 首字节: 0x5A vs 0x11 → 固件不识别 opcode")
    print(f"  4. 数据编码: 6×u8(0-255) vs 7×u16_LE(0-65535)")
    print(f"  5. 通道数: 6 vs 7 (缺少 thumb_tendon)")
    print("=" * 70 + "\n")


def run_proxy(real_port: str, baudrate: int):
    """创建虚拟串口对，代理所有流量并打印。"""
    # 创建 pty pair
    master, slave = os.openpty()
    slave_name = os.ttyname(slave)
    print(f"[sniffer] 虚拟串口已创建: {slave_name}")
    print(f"[sniffer] 请在 GUI 中选择此端口: {slave_name}")
    print(f"[sniffer] 真实串口: {real_port} @ {baudrate}")
    print(f"[sniffer] 等待数据...\n")

    real = serial.Serial(real_port, baudrate, timeout=0)

    tx_buf = bytearray()
    rx_buf = bytearray()

    try:
        while True:
            # 使用 select 同时监听两个方向
            ready, _, _ = select.select([master, real.fd], [], [], 0.01)

            # GUI → 手 (TX)
            if master in ready:
                data = os.read(master, 4096)
                if data:
                    real.write(data)
                    tx_buf.extend(data)
                    # 尝试解析完整帧
                    while len(tx_buf) >= 16:
                        frame = bytes(tx_buf[:16])
                        print(parse_frame(frame, "TX→手"))
                        print()
                        tx_buf = tx_buf[16:]

            # 手 → GUI (RX)
            if real.fd in ready:
                data = real.read(4096)
                if data:
                    os.write(master, data)
                    rx_buf.extend(data)
                    while len(rx_buf) >= 16:
                        frame = bytes(rx_buf[:16])
                        print(parse_frame(frame, "手→RX"))
                        print()
                        rx_buf = rx_buf[16:]

    except KeyboardInterrupt:
        print("\n[sniffer] 用户中断")
    finally:
        real.close()
        os.close(master)
        os.close(slave)
        print("[sniffer] 已关闭")


def run_dump(real_port: str, baudrate: int):
    """直接监听串口（只读），打印收到的帧。"""
    print(f"[sniffer] 监听模式: {real_port} @ {baudrate}")
    print(f"[sniffer] 等待数据... (Ctrl+C 退出)\n")

    real = serial.Serial(real_port, baudrate, timeout=0.1)
    buf = bytearray()

    try:
        while True:
            data = real.read(256)
            if data:
                buf.extend(data)
                while len(buf) >= 16:
                    frame = bytes(buf[:16])
                    print(parse_frame(frame, "RX"))
                    print()
                    buf = buf[16:]
    except KeyboardInterrupt:
        print("\n[sniffer] 用户中断")
    finally:
        real.close()


def main():
    parser = argparse.ArgumentParser(description="串口嗅探/代理工具")
    parser.add_argument("--real", default="/dev/ttyUSB0", help="真实串口设备")
    parser.add_argument("--baudrate", type=int, default=921600, help="波特率")
    parser.add_argument("--dump", action="store_true",
                        help="直接监听模式（不创建代理）")
    parser.add_argument("--compare", action="store_true",
                        help="只打印旧/新协议字节对比，不连接硬件")
    args = parser.parse_args()

    if args.compare:
        print_our_fist_comparison()
        return

    if args.dump:
        run_dump(args.real, args.baudrate)
    else:
        run_proxy(args.real, args.baudrate)


if __name__ == "__main__":
    main()

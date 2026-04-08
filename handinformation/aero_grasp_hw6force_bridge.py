"""AeroCubeGraspHW6ForceV 策略 -> 实物手串口桥接脚本。

本脚本对接 HW6Force 六通道力感知模型（V-iter30），特点：
1. 六路动作直连硬件六通道，无需 7→6 融合。
2. 策略观测 = [hw_pos(6), tactile_zeros(5), last_act(6)] = 17 维。
3. 力反馈在训练中仅用于奖励，策略观测中为全零——硬件也无需回传力到策略。

使用方式：
  conda activate aero_rl
  python aero_grasp_hw6force_bridge.py --mode rl            # RL 推理
  python aero_grasp_hw6force_bridge.py --mode debug         # 小幅正弦扫描
  python aero_grasp_hw6force_bridge.py --mode observe       # 只读反馈不控制

运行参数已内置合理默认值（含 checkpoint 路径），VS Code 内可直接 F5 执行。
"""

import argparse
import csv
import functools
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jp
import numpy as np
import serial

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train
from etils import epath

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params

# ──────────────────── 默认配置（VS Code 一键运行） ──────────────────── #
DEFAULT_CHECKPOINT_PATH = (
    "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/logs/"
    "AeroCubeGraspHW6ForceV-20260406-035526-v_R30_lr5e5_from41M/"
    "checkpoints/000013107200"
)
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_ENV_NAME = "AeroCubeGraspHW6ForceV"
DEFAULT_CONTROL_DT = 0.05        # 20 Hz，与训练一致
DEFAULT_MAX_TARGET_DELTA = 0.03   # 每步最大 ctrl 变化量（限速保护）
DEFAULT_LOG_CSV = "/home/ll/SRTP/Aero-Hand/handinformation/aero_bridge_hw6force_runlog.csv"
DEFAULT_PREOPEN_S = 1.2           # 启动前开手持续时间
DEFAULT_ACTION_GAIN = 1.0         # 默认 1.0 即使用训练时原始幅度

# ──── 物理单位常量 ──── #
# 训练 ctrl 空间单位：ch0=弧度, ch1-5=米（腱长）
# 硬件协议：0x5A 帧，115200 波特率，6 通道 [使能, 字节值]
#   字节值 0 = 全开, 255 = 最大弯曲
# 转换关系：ctrl → deg(0-90) → byte(0-255)
CTRL_RANGES = np.array([
    [-0.1000, 1.7500],   # hw_thumb_rot (joint, radians)
    [ 0.0816, 0.1121],   # hw_thumb_flex (tendon, meters)
    [ 0.0585, 0.1104],   # hw_index
    [ 0.0585, 0.1104],   # hw_middle
    [ 0.0585, 0.1104],   # hw_ring
    [ 0.0585, 0.1104],   # hw_pinky
], dtype=np.float32)

# 默认肌腱基线 = home keyframe ctrl（open_ratio=0.0 时）
DEFAULT_TENDON = np.array(
    [0.0, 0.1099, 0.1074, 0.1081, 0.1074, 0.1093], dtype=np.float32
)
ACTION_SCALE = np.array(
    [1.3, 0.03, 0.055, 0.055, 0.055, 0.055], dtype=np.float32
)

# 硬件标定系数（calib_hw6_full.csv 拟合结果）
# readback_deg ≈ k * cmd_deg + b
HW6_K = np.array([0.9933, 0.9928, 1.0018, 1.0029, 1.0004, 1.0002], dtype=np.float32)
HW6_B = np.array([-0.2371, -0.5978, -0.2071, -0.2094, -0.2409, -0.0640], dtype=np.float32)

# 触觉传感器加权池化系数（4×4 中心加权，与训练环境 grasp_cube_hw6_force.py 一致）
TAXEL_WEIGHTS = np.array([
    0.7, 1.0, 1.0, 0.7,
    1.0, 1.4, 1.4, 1.0,
    1.0, 1.4, 1.4, 1.0,
    0.7, 1.0, 1.0, 0.7,
], dtype=np.float32)

DEFAULT_FORCE_SAFETY_THRESHOLD = 0  # 0=禁用; >0 时触觉超阈值则衰减动作


def log(msg: str) -> None:
    print(f"[hw6force_bridge] {msg}", flush=True)


# ────────────────────── ctrl ↔ deg 转换 ────────────────────── #
def ctrl_to_deg(ctrl6: np.ndarray) -> np.ndarray:
    """将 6 维策略 ctrl 值转换为硬件 0~90 度角度。"""
    deg = np.zeros(6, dtype=np.float32)
    # ch0 (thumb_rot): joint → rad → deg
    deg[0] = np.clip(ctrl6[0], 0.0, None) * (180.0 / np.pi)
    # ch1-5 (tendons): ctrl_hi 对应 0°(开), ctrl_lo 对应 90°(闭)
    for i in range(1, 6):
        lo, hi = CTRL_RANGES[i]
        rng = hi - lo
        if rng > 1e-8:
            deg[i] = (hi - ctrl6[i]) / rng * 90.0
        else:
            deg[i] = 0.0
    return np.clip(deg, 0.0, 90.0)


def deg_to_ctrl(deg6: np.ndarray) -> np.ndarray:
    """将 6 维硬件 0~90 度角度值转换为策略 ctrl 空间值。"""
    ctrl = np.zeros(6, dtype=np.float32)
    # ch0: deg → rad
    ctrl[0] = np.clip(deg6[0], 0.0, 90.0) * (np.pi / 180.0)
    # ch1-5
    for i in range(1, 6):
        lo, hi = CTRL_RANGES[i]
        rng = hi - lo
        ctrl[i] = hi - (np.clip(deg6[i], 0.0, 90.0) / 90.0) * rng
    return ctrl


# ────────────────────── 策略加载 ────────────────────── #
def load_policy(env_name: str, checkpoint_path: str):
    """加载 PPO 策略。

    使用 ppo_train(num_timesteps=0, restore_checkpoint_path=...) 方式，
    与训练和 sim2sim 部署脚本一致，确保 Orbax checkpoint 正确加载。
    """
    env = registry.load(env_name)
    ppo_params = manipulation_params.brax_ppo_config(env_name)

    network_factory_config = ppo_params.get("network_factory", {})
    # 清理 train-only 参数
    for k in ("num_timesteps", "network_factory"):
        if k in ppo_params:
            del ppo_params[k]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_factory_config
    )

    make_inference_fn, params, _ = ppo_train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        num_timesteps=0,
        seed=1,
        restore_checkpoint_path=epath.Path(checkpoint_path).resolve(),
        **ppo_params,
    )

    if isinstance(params, dict):
        normalizer = params.get("normalizer") or params.get("normalizer_params")
        policy = params.get("policy") or params.get("policy_params") or params.get("params")
        if normalizer is not None and policy is not None:
            params = (normalizer, policy)

    jit_inference = jax.jit(make_inference_fn(params, deterministic=True))

    # 预热 JIT
    log("JIT warmup ...")
    dummy_obs = {"state": np.zeros(17, dtype=np.float32)}
    rng = jax.random.PRNGKey(0)
    _ = jit_inference(dummy_obs, rng)
    log("JIT warmup done")

    return jit_inference


# ────────────────────── 串口协议 ────────────────────── #
class HandSerialProtocol:
    """串口协议封装（动作控制 + 角度查询 + 触觉读取 + 力控配置）。

    触觉传感器数据由灵巧手主动上传（无需查询），每指一帧 64 字节，
    包含 16 个 uint16 taxel 值。后台线程持续解析并存储最新数据。
    """

    START = 0x5A
    END = 0x5D
    CMD_QUERY_ANGLE = 0xF1
    LEN_QUERY = 0x07
    TACTILE_FRAME_LEN = 64
    ANGLE_REPLY_LEN = 12

    def __init__(self, port: str, baudrate: int, timeout: float = 0.005):
        self.ser = serial.Serial(port, baudrate)
        self.ser.timeout = timeout
        self._last_query_angles_deg: Optional[np.ndarray] = None
        self._query_ok = 0
        self._query_fail = 0

        # 触觉传感器：5 指 × 16 taxels (uint16)
        self._tactile_raw = np.zeros((5, 16), dtype=np.uint16)
        self._tactile_ts = np.zeros(5, dtype=np.float64)
        self._tactile_lock = threading.Lock()
        self._tactile_update_count = 0

        # 角度回复同步
        self._angle_reply: Optional[np.ndarray] = None
        self._angle_event = threading.Event()

        # 后台接收线程
        self._rx_running = True
        self._rx_buf = bytearray()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, daemon=True, name="serial_rx"
        )
        self._rx_thread.start()

    # ──── 后台帧解析 ──── #

    def _rx_loop(self) -> None:
        """后台线程：持续读取串口，按帧类型分发处理。"""
        while self._rx_running:
            try:
                n = self.ser.in_waiting or 1
                chunk = self.ser.read(n)
                if not chunk:
                    continue
                self._rx_buf.extend(chunk)
                self._process_rx_buf()
            except serial.SerialException:
                if self._rx_running:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.001)

    def _process_rx_buf(self) -> None:
        """从接收缓冲区解析所有完整帧（全部为 0x5A 帧）。"""
        buf = self._rx_buf
        while len(buf) >= 7:
            if buf[0] != self.START:
                buf.pop(0)
                continue

            if len(buf) < 4:
                break

            # 触觉帧: 5A [1-5] 40 83 ... (64 bytes)
            if buf[1] in (1, 2, 3, 4, 5) and buf[2] == 0x40 and buf[3] == 0x83:
                if len(buf) < self.TACTILE_FRAME_LEN:
                    break
                frame = bytes(buf[: self.TACTILE_FRAME_LEN])
                del buf[: self.TACTILE_FRAME_LEN]
                self._handle_tactile_frame(frame)
                continue

            # 角度回复帧: 5A F1 ... (12 bytes)
            if buf[1] == self.CMD_QUERY_ANGLE:
                if len(buf) < self.ANGLE_REPLY_LEN:
                    break
                frame = bytes(buf[: self.ANGLE_REPLY_LEN])
                del buf[: self.ANGLE_REPLY_LEN]
                result = self._parse_query_angle_reply(frame)
                if result is not None:
                    self._query_ok += 1
                    self._last_query_angles_deg = result.copy()
                    self._angle_reply = result
                    self._angle_event.set()
                continue

            # 力控回复帧: 5A 40/4A/4B ... (7~12 bytes)
            if buf[1] in (0x40, 0x4A, 0x4B):
                flen = 12 if buf[1] == 0x4B else 7
                if len(buf) < flen:
                    break
                del buf[:flen]
                continue

            # 无法识别，跳过当前 0x5A
            buf.pop(0)

    def _handle_tactile_frame(self, frame: bytes) -> None:
        """解析 64 字节触觉帧并存储。"""
        if frame[63] != self.END:
            return
        finger_id = frame[1]  # 1~5: 拇指/食指/中指/无名指/小指
        if finger_id < 1 or finger_id > 5:
            return
        # 校验和: D3~D61 累加和取低 8 位
        cksum = sum(frame[3:62]) & 0xFF
        if cksum != frame[62]:
            return
        # 解析 16 个 uint16 大端数据点 (D4..D35)
        taxels = np.zeros(16, dtype=np.uint16)
        for i in range(16):
            off = 4 + 2 * i
            taxels[i] = (frame[off] << 8) | frame[off + 1]
        idx = finger_id - 1
        with self._tactile_lock:
            self._tactile_raw[idx] = taxels
            self._tactile_ts[idx] = time.time()
            self._tactile_update_count += 1

    # ──── 触觉数据接口 ──── #

    def get_tactile_raw(self) -> np.ndarray:
        """原始触觉数据 (5, 16) uint16。顺序: [thumb, index, middle, ring, pinky]。"""
        with self._tactile_lock:
            return self._tactile_raw.copy()

    def get_tactile_pooled(self) -> np.ndarray:
        """加权平均后每指力值 (5,) float。权重与训练环境 taxel_weights 一致。"""
        with self._tactile_lock:
            raw = self._tactile_raw.astype(np.float32)
        return np.sum(raw * TAXEL_WEIGHTS[None, :], axis=1)

    def get_tactile_timestamps(self) -> np.ndarray:
        """每指最近一次更新时间戳 (5,)。"""
        with self._tactile_lock:
            return self._tactile_ts.copy()

    # ──── 发送工具 ──── #

    @staticmethod
    def _checksum(body: list[int]) -> int:
        return sum(body) & 0xFF

    def _send_body(self, body: list[int]) -> None:
        """0x5A 协议帧：[5A] + body + [checksum] + [5D]。"""
        packet = [self.START] + body + [self._checksum(body), self.END]
        self.ser.write(bytes(packet))

    @staticmethod
    def _deg_to_u8(angle_deg: float) -> int:
        """角度 0~90° → 协议字节 0~90（字节值=度数，协议V1.5）。"""
        a = float(np.clip(angle_deg, 0.0, 90.0))
        return int(round(a))

    def send_targets_deg(self, targets_deg6: np.ndarray) -> None:
        """发送 6 通道角度命令，全使能。"""
        self.send_targets_deg_with_enable(targets_deg6, np.ones(6, dtype=np.int32))

    def send_targets_deg_with_enable(
        self, targets_deg6: np.ndarray, enables6: np.ndarray
    ) -> None:
        """发送 6 通道角度命令，带使能位。"""
        targets_deg6 = np.asarray(targets_deg6, dtype=np.float32).reshape(6)
        enables6 = np.asarray(enables6, dtype=np.int32).reshape(6)
        enables6 = np.where(enables6 > 0, 1, 0)

        body = [0x10, 0x11]
        for i in range(6):
            body.append(int(enables6[i]))
            body.append(self._deg_to_u8(float(targets_deg6[i])))
        self._send_body(body)

    def send_open(self) -> None:
        """全开手：6 通道角度 = 0。"""
        self.send_targets_deg(np.zeros(6, dtype=np.float32))

    def send_relax(self) -> None:
        """放松：使能位全 0。"""
        self.send_targets_deg_with_enable(
            np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.int32)
        )

    def send_force_control_config(self) -> None:
        body = [
            0x40, 0x28, 0x00, 0x01, 0x64, 0x4B, 0x4B, 0x01, 0x64, 0x00,
            0x28, 0x00, 0x0A, 0x01, 0x64, 0x00, 0x28, 0x00, 0x0A, 0x01,
            0x64, 0x00, 0x5A, 0x00, 0x0A, 0x01, 0x64, 0x00, 0x5A, 0x00,
            0x0A, 0x01, 0x64, 0x00, 0x5A, 0x00, 0x0A,
        ]
        self._send_body(body)

    def set_force_mode(self, enabled: bool) -> None:
        body = [0x4A, 0x07, 0x00, 0x01 if enabled else 0x00]
        self._send_body(body)

    # ──── 角度查询（0xF1） ──── #

    def _send_query_angle(self) -> None:
        body = [self.CMD_QUERY_ANGLE, self.LEN_QUERY, 0x00, 0x00]
        self._send_body(body)

    @staticmethod
    def _parse_query_angle_reply(frame12: bytes) -> Optional[np.ndarray]:
        """解析角度回复帧 → 6 路角度（uint8 原始值，作为度数）。

        帧格式：5A F1 0C 00 a0 a1 a2 a3 a4 a5 CHK 5D
        """
        if len(frame12) != 12:
            return None
        if frame12[0] != 0x5A or frame12[1] != 0xF1 or frame12[11] != 0x5D:
            return None
        d1_to_d9 = list(frame12[1:10])
        if (sum(d1_to_d9) & 0xFF) != int(frame12[10]):
            return None
        return np.frombuffer(bytes(frame12[4:10]), dtype=np.uint8).astype(np.float32)

    def query_angles_once(self) -> Optional[np.ndarray]:
        """查询一次角度。后台线程自动接收回复，此处仅发送并等待。"""
        self._angle_event.clear()
        self._send_query_angle()
        if self._angle_event.wait(timeout=0.05):
            return self._angle_reply
        self._query_fail += 1
        return None

    def feedback_debug(self) -> dict:
        with self._tactile_lock:
            t_count = self._tactile_update_count
            t_ts = self._tactile_ts.copy()
        return {
            "source": "query" if self._last_query_angles_deg is not None else "none",
            "query_ok": int(self._query_ok),
            "query_fail": int(self._query_fail),
            "last_query_angles_deg": (
                None if self._last_query_angles_deg is None
                else self._last_query_angles_deg.copy()
            ),
            "tactile_updates": int(t_count),
            "tactile_age_s": (
                round(time.time() - float(np.max(t_ts)), 3)
                if np.any(t_ts > 0) else -1.0
            ),
        }

    def close(self) -> None:
        self._rx_running = False
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self.ser.is_open:
            self.ser.close()


# ────────────────────── 硬件桥接核心 ────────────────────── #
class AeroHW6ForceBridge:
    """HW6Force 策略到硬件的实时桥接。

    闭环流程：读角度反馈 → 构造 17D obs → 策略推理 → ctrl → deg → 串口下发。
    """

    def __init__(
        self,
        checkpoint_path: str,
        serial_port: str,
        serial_baudrate: int,
        env_name: str = DEFAULT_ENV_NAME,
        control_dt: float = DEFAULT_CONTROL_DT,
        max_target_delta: float = DEFAULT_MAX_TARGET_DELTA,
        verbose: bool = True,
        heartbeat_steps: int = 20,
        control_mode: str = "rl",
        debug_amp_deg: float = 8.0,
        debug_period_s: float = 2.5,
        max_steps: int = 0,
        log_csv_path: Optional[str] = DEFAULT_LOG_CSV,
        preopen_s: float = DEFAULT_PREOPEN_S,
        action_gain: float = DEFAULT_ACTION_GAIN,
        force_safety_threshold: float = 0.0,
    ):
        self.verbose = bool(verbose)
        self.heartbeat_steps = int(max(1, heartbeat_steps))
        self.control_mode = str(control_mode).lower()
        self.debug_amp_deg = float(max(0.0, debug_amp_deg))
        self.debug_period_s = float(max(0.2, debug_period_s))
        self.max_steps = int(max(0, max_steps))
        self.preopen_s = float(max(0.0, preopen_s))
        self.action_gain = float(np.clip(action_gain, 0.0, 2.0))
        self.force_safety_threshold = float(max(0.0, force_safety_threshold))
        self.control_dt = float(control_dt)
        self.max_target_delta = float(max_target_delta)
        self._step_idx = 0

        # ── 策略加载 ── #
        if self.control_mode == "rl":
            if self.verbose:
                log(f"loading policy: {checkpoint_path}")
            self.policy_fn = load_policy(env_name, checkpoint_path)
            if self.verbose:
                log(f"policy ready (action_gain={self.action_gain:.2f})")
        elif self.control_mode == "observe":
            self.policy_fn = None
            if self.verbose:
                log("observe mode: 只读反馈不发送控制命令")
        else:
            self.policy_fn = None
            if self.verbose:
                log(f"debug mode: amp={self.debug_amp_deg:.1f}° period={self.debug_period_s:.1f}s")

        self.rng = jax.random.PRNGKey(0)

        # ── 串口 ── #
        if self.verbose:
            log(f"opening serial: {serial_port} @ {serial_baudrate}")
        self.hand = HandSerialProtocol(serial_port, serial_baudrate)
        if self.verbose:
            log("serial opened")

        # ── 状态缓存 ── #
        self.default_tendon = DEFAULT_TENDON.copy()
        self.action_scale = ACTION_SCALE.copy()
        self.last_action = jp.zeros(6, dtype=jp.float32)
        self.last_action_np = np.zeros(6, dtype=np.float32)
        self.last_raw_action_np = np.zeros(6, dtype=np.float32)
        # motor_targets 在 ctrl 空间（与训练一致）
        self.last_motor_targets = self.default_tendon.copy()
        # 最近一次下发的角度（度）
        self.last_cmd_deg = ctrl_to_deg(self.default_tendon)
        # 最近一次回读的角度（度），用于 CSV
        self.last_readback_deg = np.zeros(6, dtype=np.float32)
        # 触觉
        self.last_tactile_pooled = np.zeros(5, dtype=np.float32)
        self.last_tactile_raw = np.zeros((5, 16), dtype=np.uint16)
        self.last_force_safety_scale = 1.0

        # ── CSV ── #
        self.log_csv_path = log_csv_path
        self._csv_file = None
        self._csv_writer = None
        if self.log_csv_path:
            p = Path(self.log_csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = p.open("w", newline="", encoding="utf-8")
            fields = (
                ["time", "step", "mode"]
                + [f"cmd_deg_{i}" for i in range(6)]
                + [f"motor_target_{i}" for i in range(6)]
                + ["action_norm"]
                + [f"raw_action_{i}" for i in range(6)]
                + [f"obs_hw_pos_{i}" for i in range(6)]
                + ["fb_source", "q_ok", "q_fail"]
                + [f"readback_deg_{i}" for i in range(6)]
                + [f"tactile_pooled_{i}" for i in range(5)]
                + ["force_safety_scale"]
                + [f"tactile_raw_{f}_{t}" for f in range(5) for t in range(16)]
            )
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fields)
            self._csv_writer.writeheader()
            if self.verbose:
                log(f"csv logging: {p}")

    # ──── 硬件反馈 → 策略 ctrl 空间 ──── #

    def _read_hw_pos(self) -> np.ndarray:
        """读取硬件角度反馈，转换为 ctrl 空间的 hw_pos (6D)。

        失败时回退到上次的 motor_targets（与训练中 hw_pos=motor_targets 一致）。
        """
        angles_deg = self.hand.query_angles_once()
        if angles_deg is None:
            return self.last_motor_targets.copy()

        self.last_readback_deg = angles_deg.copy()

        # 按标定反解：eq_cmd_deg = (readback_deg - b) / k
        safe_k = np.where(np.abs(HW6_K) < 1e-6, 1.0, HW6_K)
        eq_cmd_deg = (angles_deg - HW6_B) / safe_k
        eq_cmd_deg = np.clip(eq_cmd_deg, 0.0, 90.0)

        # deg → ctrl 空间
        return deg_to_ctrl(eq_cmd_deg)

    # ──── 构造策略观测 ──── #

    def _build_policy_obs(self) -> dict:
        """构造 17D 策略输入:  [hw_pos(6), tactile_zeros(5), last_act(6)]。

        hw_pos 使用 last_motor_targets（命令值）而非硬件实际反馈，
        与训练环境一致（训练中 hw_pos = info["motor_targets"]，即控制目标）。
        真实触觉数据保存在 self.last_tactile_* 供日志和安全层使用。
        """
        # 真实读回仅用于日志和标定，不进策略观测
        self._read_hw_pos()
        # hw_pos = 命令值（与训练一致：sim中 hw_pos = motor_targets）
        hw_pos = self.last_motor_targets.copy()
        # 读取真实触觉（用于日志和力安全，不进策略）
        self.last_tactile_pooled = self.hand.get_tactile_pooled()
        self.last_tactile_raw = self.hand.get_tactile_raw()
        # 策略 obs 中 tactile 仍为零（兼容已训练模型的 normalizer）
        tactile_obs = np.zeros(5, dtype=np.float32)
        state = np.concatenate([hw_pos, tactile_obs, np.asarray(self.last_action)])
        return {"state": state.astype(np.float32)}

    # ──── 限速 ──── #

    def _rate_limit(self, targets: np.ndarray) -> np.ndarray:
        delta = np.clip(
            targets - self.last_motor_targets,
            -self.max_target_delta,
            self.max_target_delta,
        )
        return self.last_motor_targets + delta

    # ──── debug 正弦扫描 ──── #

    def _debug_targets_deg(self) -> np.ndarray:
        t = self._step_idx * self.control_dt
        phase = (2.0 * np.pi * t) / self.debug_period_s
        base_deg = ctrl_to_deg(self.default_tendon)
        deg = base_deg.copy()
        for i in range(6):
            wave = np.sin(phase + i * (np.pi / 3.0))
            deg[i] = base_deg[i] + self.debug_amp_deg * wave
        return np.clip(deg, 0.0, 90.0)

    # ──── 单步闭环 ──── #

    def step_once(self) -> None:
        if self.control_mode == "rl":
            obs = self._build_policy_obs()
            self.rng, act_rng = jax.random.split(self.rng)
            action_jax, _ = self.policy_fn(obs, act_rng)
            action = np.asarray(action_jax, dtype=np.float32).reshape(6)

            self.last_raw_action_np = action.copy()
            action_scaled = action * self.action_gain

            # ── 力安全层：触觉超阈值时衰减动作 ── #
            self.last_force_safety_scale = 1.0
            if self.force_safety_threshold > 0:
                max_force = float(np.max(self.last_tactile_pooled))
                if max_force > self.force_safety_threshold:
                    self.last_force_safety_scale = float(np.clip(
                        1.0 - (max_force - self.force_safety_threshold)
                        / self.force_safety_threshold, 0.0, 1.0
                    ))
                    action_scaled = action_scaled * self.last_force_safety_scale

            motor_targets = self.default_tendon + action_scaled * self.action_scale

            # clip 到 ctrl 物理范围
            motor_targets = np.clip(
                motor_targets, CTRL_RANGES[:, 0], CTRL_RANGES[:, 1]
            )
            motor_targets = self._rate_limit(motor_targets)

            self.last_action = jp.array(action_scaled, dtype=jp.float32)
            self.last_action_np = action_scaled.copy()
            self.last_motor_targets = motor_targets.copy()

            # 转为角度下发
            cmd_deg = ctrl_to_deg(motor_targets)
            # 再过一次标定反解: cmd_real = (desired_deg - b) / k
            safe_k = np.where(np.abs(HW6_K) < 1e-6, 1.0, HW6_K)
            cmd_deg_calib = (cmd_deg - HW6_B) / safe_k
            cmd_deg_calib = np.clip(cmd_deg_calib, 0.0, 90.0)
            self.last_cmd_deg = cmd_deg_calib.copy()
            self.hand.send_targets_deg(cmd_deg_calib)

        elif self.control_mode == "observe":
            _ = self._build_policy_obs()
            self.last_action = jp.zeros(6, dtype=jp.float32)
            self.last_action_np = np.zeros(6, dtype=np.float32)
            self.last_raw_action_np = np.zeros(6, dtype=np.float32)
            self._step_idx += 1
            return

        else:  # debug
            cmd_deg = self._debug_targets_deg()
            self.last_cmd_deg = cmd_deg.copy()
            self.hand.send_targets_deg(cmd_deg)
            self.last_motor_targets = deg_to_ctrl(cmd_deg)
            self.last_action = jp.zeros(6, dtype=jp.float32)
            self.last_action_np = np.zeros(6, dtype=np.float32)
            self.last_raw_action_np = np.zeros(6, dtype=np.float32)

        self._step_idx += 1

    # ──── CSV 日志 ──── #

    def _write_csv_row(self, step_count: int) -> None:
        if self._csv_writer is None:
            return
        fb_dbg = self.hand.feedback_debug()
        rb = self.last_readback_deg
        row = {
            "time": time.time(),
            "step": int(step_count),
            "mode": self.control_mode,
            "action_norm": float(np.linalg.norm(self.last_action_np)),
            "fb_source": fb_dbg["source"],
            "q_ok": int(fb_dbg["query_ok"]),
            "q_fail": int(fb_dbg["query_fail"]),
        }
        for i in range(6):
            row[f"cmd_deg_{i}"] = float(self.last_cmd_deg[i])
            row[f"motor_target_{i}"] = float(self.last_motor_targets[i])
            row[f"raw_action_{i}"] = float(self.last_raw_action_np[i])
            row[f"obs_hw_pos_{i}"] = float(self.last_motor_targets[i])
            row[f"readback_deg_{i}"] = float(rb[i])
        for i in range(5):
            row[f"tactile_pooled_{i}"] = float(self.last_tactile_pooled[i])
        row["force_safety_scale"] = float(self.last_force_safety_scale)
        for f in range(5):
            for t in range(16):
                row[f"tactile_raw_{f}_{t}"] = int(self.last_tactile_raw[f, t])
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    # ──── 主循环 ──── #

    def run(self) -> None:
        next_t = time.perf_counter()
        step_count = 0

        try:
            # 启动前开手
            if self.control_mode in ("rl", "debug") and self.preopen_s > 0.0:
                preopen_n = max(1, int(round(self.preopen_s / max(self.control_dt, 1e-3))))
                for _ in range(preopen_n):
                    self.hand.send_open()
                    time.sleep(self.control_dt)
                if self.verbose:
                    log(f"preopen done ({self.preopen_s:.1f}s)")

            if self.control_mode == "observe":
                # 放松模式
                self.hand.set_force_mode(False)
                time.sleep(0.05)
                preopen_n = max(1, int(round(self.preopen_s / max(self.control_dt, 1e-3))))
                for _ in range(preopen_n):
                    self.hand.send_open()
                    time.sleep(self.control_dt)
                self.hand.send_relax()
                time.sleep(0.05)
                if self.verbose:
                    log("observe mode: force_off + preopen + relax")

            if self.verbose:
                log(f"control loop started (dt={self.control_dt:.3f}s)")

            while True:
                self.step_once()
                step_count += 1
                self._write_csv_row(step_count)

                if self.verbose and step_count % self.heartbeat_steps == 0:
                    fb_dbg = self.hand.feedback_debug()
                    anorm = float(np.linalg.norm(self.last_action_np))
                    qangles_txt = (
                        "none"
                        if fb_dbg["last_query_angles_deg"] is None
                        else np.round(fb_dbg["last_query_angles_deg"], 1).tolist()
                    )
                    tp = np.round(self.last_tactile_pooled, 1).tolist()
                    log(
                        f"step={step_count} "
                        f"cmd_deg={np.round(self.last_cmd_deg, 1).tolist()} "
                        f"action_norm={anorm:.4f} "
                        f"q_ok={fb_dbg['query_ok']} "
                        f"q_fail={fb_dbg['query_fail']} "
                        f"readback_deg={qangles_txt} "
                        f"tactile_pooled={tp} "
                        f"tactile_updates={fb_dbg['tactile_updates']} "
                        f"force_safety={self.last_force_safety_scale:.2f}"
                    )

                next_t += self.control_dt
                sleep_t = next_t - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.perf_counter()

                if self.max_steps > 0 and step_count >= self.max_steps:
                    if self.verbose:
                        log(f"max_steps reached ({self.max_steps})")
                    break

        finally:
            self.hand.close()
            if self._csv_file is not None:
                self._csv_file.close()
                if self.verbose:
                    log("csv closed")


def parse_args():
    p = argparse.ArgumentParser(
        description="AeroCubeGraspHW6ForceV RL policy → hardware bridge"
    )
    p.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--serial_port", default=DEFAULT_SERIAL_PORT)
    p.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    p.add_argument("--env_name", default=DEFAULT_ENV_NAME)
    p.add_argument("--control_dt", type=float, default=DEFAULT_CONTROL_DT)
    p.add_argument("--max_target_delta", type=float, default=DEFAULT_MAX_TARGET_DELTA)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--heartbeat_steps", type=int, default=20)
    p.add_argument(
        "--mode",
        choices=["rl", "debug", "observe"],
        default="rl",
    )
    p.add_argument("--debug_amp_deg", type=float, default=8.0)
    p.add_argument("--debug_period_s", type=float, default=2.5)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--log_csv", default=DEFAULT_LOG_CSV)
    p.add_argument("--preopen_s", type=float, default=DEFAULT_PREOPEN_S)
    p.add_argument("--action_gain", type=float, default=DEFAULT_ACTION_GAIN)
    p.add_argument(
        "--force_safety_threshold", type=float,
        default=DEFAULT_FORCE_SAFETY_THRESHOLD,
        help="触觉池化值超此阈值时衰减动作 (0=禁用; 建议先 observe 模式观察典型值后设定)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    log("=== AeroCubeGraspHW6ForceV Hardware Bridge ===")
    log(f"mode={args.mode}")
    log(f"checkpoint={args.checkpoint_path}")
    log(f"serial={args.serial_port} baud={args.baudrate}")
    log(f"action_gain={args.action_gain}")
    log(f"force_safety_threshold={args.force_safety_threshold}")

    if args.mode == "rl":
        ckpt = Path(args.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"checkpoint 路径不存在: {ckpt}\n"
                "请修改脚本顶部 DEFAULT_CHECKPOINT_PATH 或通过 --checkpoint_path 指定。"
            )

    bridge = AeroHW6ForceBridge(
        checkpoint_path=args.checkpoint_path,
        serial_port=args.serial_port,
        serial_baudrate=args.baudrate,
        env_name=args.env_name,
        control_dt=args.control_dt,
        max_target_delta=args.max_target_delta,
        verbose=not args.quiet,
        heartbeat_steps=args.heartbeat_steps,
        control_mode=args.mode,
        debug_amp_deg=args.debug_amp_deg,
        debug_period_s=args.debug_period_s,
        max_steps=args.max_steps,
        log_csv_path=(args.log_csv.strip() if args.log_csv else ""),
        preopen_s=args.preopen_s,
        action_gain=args.action_gain,
        force_safety_threshold=args.force_safety_threshold,
    )
    bridge.run()


if __name__ == "__main__":
    main()

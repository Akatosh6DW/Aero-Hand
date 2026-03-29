"""Aero 抓取策略 -> 实物手串口桥接脚本。

本文件的目标：
1. 复用仿真策略（AeroCubeGrasp checkpoint）。
2. 在固定控制周期内读取硬件反馈，拼装策略观测。
3. 输出连续控制命令到串口协议（而非固定手势）。

和仿真脚本的对接关系：
- 仿真端参考：sim_rl/mujoco_playground/mujoco_playground/experimental/sim2sim/play_aero_grasp_jax.py
- 两边保持一致的关键是：state 观测结构、action 映射、控制频率。
"""

import argparse
import csv
import functools
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


# VS Code 一键运行默认配置：
# 1) 首次使用只需要改 DEFAULT_CHECKPOINT_PATH。
# 2) 其余参数可按硬件实际再微调。
DEFAULT_CHECKPOINT_PATH = "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/logs/AeroCubeGrasp-20260321-005730/checkpoints/000893255680"
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_ENV_NAME = "AeroCubeGrasp"
DEFAULT_HISTORY_LEN = 1
DEFAULT_CONTROL_DT = 0.05
DEFAULT_MAX_TARGET_DELTA = 0.04
DEFAULT_LOG_CSV = "/home/ll/SRTP/Aero-Hand/handinformation/aero_bridge_runlog.csv"
DEFAULT_OBSERVE_PREOPEN_S = 1.2
DEFAULT_RL_PREOPEN_S = 1.2
DEFAULT_ACTION_GAIN = 0.35
DEFAULT_THUMB_FLEX_W4 = 0.30
DEFAULT_THUMB_FLEX_W5 = 0.70
DEFAULT_THUMB_ROT_W4 = 0.20
DEFAULT_THUMB_ROT_W6 = 0.80

# 观测重对齐参数（先用当前实测中立位，后续可按你补充的数据更新）
# 顺序（协议 0xF1）：[拇指旋转, 拇指弯曲, 食指, 中指, 无名指, 小指]
OBS_NEUTRAL_ANGLES_DEG = np.asarray([24.0, 18.0, 19.0, 18.0, 17.0, 17.0], dtype=np.float32)
OBS_ALIGN_GAIN = 1.0


def log(msg: str) -> None:
    print(f"[aero_bridge] {msg}", flush=True)


def load_policy(env_name: str, checkpoint_path: str):
    # 与仿真脚本一致：按 env 配置重建网络结构，再从 checkpoint 恢复参数。
    env = registry.load(env_name)
    ppo_params = manipulation_params.brax_ppo_config(env_name)
    network_factory_config = ppo_params.get("network_factory", {})
    del ppo_params["network_factory"]

    if "num_timesteps" in ppo_params:
        del ppo_params["num_timesteps"]

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

    return jax.jit(make_inference_fn(params, deterministic=True))


class HandSerialProtocol:
    """Serial protocol wrapper based on the vendor hand.py packet format.

    Notes:
    - Gesture packet body is [0x10, 0x11] + 12 bytes (6 channels x [0x01, value]).
    - This class sends continuous commands at runtime instead of fixed gestures.
    - Feedback parser is optional because protocol details vary by firmware.
    """

    START = 0x5A
    END = 0x5D
    CMD_QUERY_ANGLE = 0xF1
    LEN_QUERY = 0x07

    def __init__(
        self,
        port: str,
        baudrate: int,
        timeout: float = 0.005,
        feedback_parser: Optional[Callable[[bytes], Optional[dict]]] = None,
    ):
        # port: Linux 常见 /dev/ttyUSB0 或 /dev/ttyACM0。
        # baudrate: 必须与手固件设置一致，默认 115200。
        # timeout: 读串口超时，过大将拖慢控制闭环。
        # feedback_parser: 可插入厂商反馈解析器，把字节流解析成观测字典。
        self.ser = serial.Serial(port, baudrate)
        self.ser.timeout = timeout
        self.feedback_parser = feedback_parser

        self._last_targets6 = np.zeros(6, dtype=np.float32)
        self._last_feedback = None
        self._feedback_source = "none"
        self._last_query_angles_deg = None
        self._query_ok = 0
        self._query_fail = 0

    @staticmethod
    def _checksum(body: list[int]) -> int:
        return sum(body) & 0xFF

    def _send_body(self, body: list[int]) -> None:
        packet = [self.START] + body + [self._checksum(body), self.END]
        self.ser.write(bytes(packet))

    @staticmethod
    def _to_u8(v01: float) -> int:
        v01 = float(np.clip(v01, 0.0, 1.0))
        return int(round(v01 * 255.0))

    def send_targets6(self, targets6: np.ndarray) -> None:
        # targets6 为 6 路归一化目标（0~1），最终编码成 0~255 下发。
        self.send_targets6_with_enable(targets6, enables6=np.ones(6, dtype=np.int32))

    def send_targets6_with_enable(self, targets6: np.ndarray, enables6: np.ndarray) -> None:
        # enables6 为每一路使能位：1=执行，0=放松（具体效果取决于固件实现）。
        targets6 = np.asarray(targets6, dtype=np.float32).reshape(6)
        enables6 = np.asarray(enables6, dtype=np.int32).reshape(6)
        enables6 = np.where(enables6 > 0, 1, 0)
        values = [self._to_u8(v) for v in targets6]

        body = [0x10, 0x11]
        for en, val in zip(enables6.tolist(), values):
            body.extend([int(en), int(val)])

        self._send_body(body)
        self._last_targets6 = targets6.copy()

    def send_relax6(self) -> None:
        # 尝试让各通道进入“非位置保持”状态，便于手动扰动观测。
        self.send_targets6_with_enable(np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.int32))

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

    def _send_query_angle(self) -> None:
        body = [self.CMD_QUERY_ANGLE, self.LEN_QUERY, 0x00, 0x00]
        packet = [self.START] + body + [self._checksum(body), self.END]
        self.ser.write(bytes(packet))

    def _read_response_frame(self, expected_len: int = 12, timeout_s: float = 0.03) -> Optional[bytes]:
        t0 = time.time()
        buf = bytearray()
        while time.time() - t0 < timeout_s:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if not chunk:
                continue
            buf.extend(chunk)
            while len(buf) >= expected_len:
                if buf[0] != self.START:
                    buf.pop(0)
                    continue
                frame = bytes(buf[:expected_len])
                if frame[-1] == self.END:
                    return frame
                buf.pop(0)
        return None

    @staticmethod
    def _parse_query_angle_reply(frame12: bytes) -> Optional[np.ndarray]:
        # 预期帧：5A F1 0C 00 a0 a1 a2 a3 a4 a5 CHK 5D
        if len(frame12) != 12:
            return None
        if frame12[0] != 0x5A or frame12[1] != 0xF1 or frame12[11] != 0x5D:
            return None
        d1_to_d9 = list(frame12[1:10])
        if (sum(d1_to_d9) & 0xFF) != int(frame12[10]):
            return None
        return np.frombuffer(bytes(frame12[4:10]), dtype=np.uint8).astype(np.float32)

    def query_angles_once(self) -> Optional[np.ndarray]:
        # 清掉积压字节，避免旧帧干扰当前查询。
        if self.ser.in_waiting > 0:
            _ = self.ser.read(self.ser.in_waiting)

        self._send_query_angle()
        frame = self._read_response_frame(expected_len=12, timeout_s=0.03)
        if frame is None:
            return None
        return self._parse_query_angle_reply(frame)

    def read_feedback(self) -> Optional[dict]:
        # 优先走 0xF1 实时角度查询，形成真实反馈闭环。
        # 角度顺序（协议）：[拇指旋转, 拇指弯曲, 食指, 中指, 无名指, 小指]
        angles_deg = self.query_angles_once()
        if angles_deg is not None:
            self._query_ok += 1
            self._last_query_angles_deg = angles_deg.copy()
            self._feedback_source = "query"
            thumb_rot, thumb_flex, index, middle, ring, pinky = angles_deg.tolist()

            # 映射到策略观测的 6 路 tendon 语义（归一化 0~1）：
            # [if, mf, rf, pf, th1, th2] ~= [index, middle, ring, pinky, thumb_flex, thumb_rot]
            tendon = np.asarray([index, middle, ring, pinky, thumb_flex, thumb_rot], dtype=np.float32)
            tendon = np.clip(tendon / 90.0, 0.0, 1.0)

            # thumb_abd 先用拇指旋转通道近似。
            thumb_abd = float(np.clip(thumb_rot / 90.0, 0.0, 1.0))

            self._last_feedback = {
                "tendon_lengths": tendon,
                "thumb_abd": thumb_abd,
            }
            return self._last_feedback

        self._query_fail += 1

        if self.feedback_parser is None:
            # 查询失败时回退为上次下发值，避免控制中断。
            self._feedback_source = "fallback"
            self._last_feedback = {
                "tendon_lengths": self._last_targets6.copy(),
                "thumb_abd": float(self._last_targets6[4]),
            }
            return self._last_feedback

        if self.ser.in_waiting <= 0:
            return self._last_feedback

        raw = self.ser.read(self.ser.in_waiting)
        if not raw:
            return self._last_feedback

        parsed = self.feedback_parser(raw)
        if parsed is not None:
            self._feedback_source = "parser"
            self._last_feedback = parsed
        return self._last_feedback

    def feedback_debug(self) -> dict:
        return {
            "source": self._feedback_source,
            "query_ok": int(self._query_ok),
            "query_fail": int(self._query_fail),
            "last_query_angles_deg": None if self._last_query_angles_deg is None else self._last_query_angles_deg.copy(),
        }

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()


class AeroHardwareBridge:
    """Bridge: PPO policy -> hardware command packets at fixed control rate."""

    def __init__(
        self,
        checkpoint_path: str,
        serial_port: str,
        serial_baudrate: int,
        env_name: str = "AeroCubeGrasp",
        history_len: int = 1,
        control_dt: float = 0.05,
        max_target_delta: float = 0.04,
        channel_map_7_to_6: Optional[list[int]] = None,
        verbose: bool = True,
        heartbeat_steps: int = 20,
        control_mode: str = "rl",
        debug_amp_deg: float = 8.0,
        debug_period_s: float = 2.5,
        max_steps: int = 0,
        log_csv_path: Optional[str] = DEFAULT_LOG_CSV,
        observe_preopen_s: float = DEFAULT_OBSERVE_PREOPEN_S,
        rl_preopen_s: float = DEFAULT_RL_PREOPEN_S,
        action_gain: float = DEFAULT_ACTION_GAIN,
        thumb_flex_w4: float = DEFAULT_THUMB_FLEX_W4,
        thumb_flex_w5: float = DEFAULT_THUMB_FLEX_W5,
        thumb_rot_w4: float = DEFAULT_THUMB_ROT_W4,
        thumb_rot_w6: float = DEFAULT_THUMB_ROT_W6,
    ):
        self.verbose = bool(verbose)
        self.heartbeat_steps = int(max(1, heartbeat_steps))
        self.control_mode = str(control_mode).lower()
        self.debug_amp_norm = float(max(0.0, debug_amp_deg) / 90.0)
        self.debug_period_s = float(max(0.2, debug_period_s))
        self.max_steps = int(max(0, max_steps))
        self.observe_preopen_s = float(max(0.0, observe_preopen_s))
        self.rl_preopen_s = float(max(0.0, rl_preopen_s))
        self.action_gain = float(np.clip(action_gain, 0.0, 1.0))
        self.thumb_flex_w4 = float(thumb_flex_w4)
        self.thumb_flex_w5 = float(thumb_flex_w5)
        self.thumb_rot_w4 = float(thumb_rot_w4)
        self.thumb_rot_w6 = float(thumb_rot_w6)
        self._step_idx = 0
        self.log_csv_path = log_csv_path
        self._csv_file = None
        self._csv_writer = None

        # checkpoint_path: 训练输出的 Orbax checkpoint 目录。
        # env_name: 与训练环境名保持一致（默认 AeroCubeGrasp）。
        if self.control_mode == "rl":
            if self.verbose:
                log(f"loading policy from checkpoint: {checkpoint_path}")
            self.policy_fn = load_policy(env_name, checkpoint_path)
            if self.verbose:
                log(f"policy loaded (action_gain={self.action_gain:.3f})")
        elif self.control_mode == "observe":
            self.policy_fn = None
            if self.verbose:
                log("observe mode enabled: query feedback + log only, no control packets")
        else:
            self.policy_fn = None
            if self.verbose:
                log(
                    "debug mode enabled: skip policy load, "
                    f"amp={debug_amp_deg:.2f}deg period={self.debug_period_s:.2f}s"
                )
        self.rng = jax.random.PRNGKey(0)

        # 串口后端：负责协议组包与发送。
        if self.verbose:
            log(f"opening serial: {serial_port} @ {serial_baudrate}")
        self.hand = HandSerialProtocol(serial_port, serial_baudrate)
        if self.verbose:
            log("serial opened")

        # history_len: 与训练时观测历史长度一致。
        # obs_dim=14 来源：6 腱长 + 1 拇指外展 + 7 last_action。
        self.history_len = int(history_len)
        self.obs_dim = 14
        self.obs_history = jp.zeros((self.history_len * self.obs_dim,), dtype=jp.float32)
        self.last_action = jp.zeros((7,), dtype=jp.float32)
        self.last_action_np = np.zeros((7,), dtype=np.float32)
        self.last_raw_action_np = np.zeros((7,), dtype=np.float32)

        # control_dt: 控制周期（秒），建议与仿真/训练控制步保持一致（0.05s）。
        # max_target_delta: 每步最大目标变化量，用于限速，防止硬件抖动或冲击。
        self.control_dt = float(control_dt)
        self.max_target_delta = float(max_target_delta)

        # 必须与训练动作映射一致：target7 = default_tendon + action * action_scale。
        self.default_tendon = jp.array([0.083, 0.082, 0.082, 0.086, 0.75, 0.035, 0.1], dtype=jp.float32)
        self.action_scale = jp.array([0.02, 0.02, 0.02, 0.02, 0.7, 0.003, 0.012], dtype=jp.float32)

        # 硬件协议当前是 6 路，而策略动作是 7 路。
        # channel_map_7_to_6 定义了“策略7路 -> 协议6路”的映射。
        # 若映射不准，实物动作会明显偏离仿真行为。
        self.channel_map_7_to_6 = channel_map_7_to_6 or [0, 1, 2, 3, 5, 6]
        self.last_targets7 = np.asarray(self.default_tendon, dtype=np.float32)

        # 基于 handinformation/calib_hw6_full.csv（含 cycle/direction 字段）的鲁棒线性标定结果。
        # 拟合时去掉每个 sweep 的首点（首点常受上一通道残留影响）。
        # 模型为：readback_deg ~= k * cmd_deg + b
        # 桥接下发前按反函数求命令：cmd_deg = (desired_deg - b) / k
        self.hw6_k = np.asarray([0.9933, 0.9928, 1.0018, 1.0029, 1.0004, 1.0002], dtype=np.float32)
        self.hw6_b = np.asarray([-0.2371, -0.5978, -0.2071, -0.2094, -0.2409, -0.0640], dtype=np.float32)

        # 用于观测重对齐：将硬件角度映射到更接近训练 state 分布的空间。
        # 映射顺序到 6 路 tendon：[if, mf, rf, pf, th1, th2]
        self.obs_neutral_deg = OBS_NEUTRAL_ANGLES_DEG.astype(np.float32)
        self.obs_align_gain = float(OBS_ALIGN_GAIN)
        default_tendon_np = np.asarray(self.default_tendon, dtype=np.float32)
        self.default_tendon6 = default_tendon_np[[0, 1, 2, 3, 5, 6]]
        self.default_thumb_abd = float(self.default_tendon[4])
        self.last_obs_tendon6 = np.zeros(6, dtype=np.float32)
        self.last_obs_thumb_abd = 0.0
        self._baseline_q_angles = None
        self._baseline_set_step = 0
        self._baseline_warmup_steps = 20

        if self.log_csv_path:
            p = Path(self.log_csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = p.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=[
                    "time", "step", "mode",
                    "target6_0", "target6_1", "target6_2", "target6_3", "target6_4", "target6_5",
                    "target7_0", "target7_1", "target7_2", "target7_3", "target7_4", "target7_5", "target7_6",
                    "action_norm",
                    "raw_action_0", "raw_action_1", "raw_action_2", "raw_action_3", "raw_action_4", "raw_action_5", "raw_action_6",
                    "obs_tendon6_0", "obs_tendon6_1", "obs_tendon6_2", "obs_tendon6_3", "obs_tendon6_4", "obs_tendon6_5",
                    "obs_thumb_abd",
                    "fb_source", "q_ok", "q_fail",
                    "q_angle_deg_0", "q_angle_deg_1", "q_angle_deg_2", "q_angle_deg_3", "q_angle_deg_4", "q_angle_deg_5",
                    "dq_angle_deg_0", "dq_angle_deg_1", "dq_angle_deg_2", "dq_angle_deg_3", "dq_angle_deg_4", "dq_angle_deg_5",
                ],
            )
            self._csv_writer.writeheader()
            if self.verbose:
                log(f"csv logging enabled: {p}")

    def _read_feedback_state(self) -> tuple[np.ndarray, float]:
        # 输出两部分：
        # 1) tendon: 6 路腱长（if/mf/rf/pf/th1/th2）
        # 2) thumb_abd: 拇指外展量（1 路）
        # 这与仿真策略 state 的前 7 维语义对齐。
        fb = self.hand.read_feedback()
        if fb is None:
            tendon = np.asarray(self.last_targets7[[0, 1, 2, 3, 5, 6]], dtype=np.float32)
            thumb_abd = float(self.last_targets7[4])
            return tendon, thumb_abd

        # 优先使用 0xF1 原始角度做重对齐，减少观测分布偏移。
        q_angles = self.hand.feedback_debug().get("last_query_angles_deg")
        if q_angles is not None:
            q = np.asarray(q_angles, dtype=np.float32).reshape(6)

            # 先按标定反解到“等效命令角”。
            # 协议顺序 -> 校准参数顺序：
            # [thumb_rot, thumb_flex, index, middle, ring, pinky]
            # [k0, k1, k2, k3, k4, k5]
            safe_k = np.where(np.abs(self.hw6_k) < 1e-6, 1.0, self.hw6_k)
            eq_cmd_deg = (q - self.hw6_b) / safe_k

            # 中立位同样反解，做“相对中立偏差”。
            neutral_eq_deg = (self.obs_neutral_deg - self.hw6_b) / safe_k

            # 重排到策略 tendon 顺序：[if, mf, rf, pf, th1, th2]
            # from protocol index: [2,3,4,5,1,0]
            idx = [2, 3, 4, 5, 1, 0]
            eq_tendon = np.clip(eq_cmd_deg[idx] / 90.0, 0.0, 1.0)
            neutral_tendon = np.clip(neutral_eq_deg[idx] / 90.0, 0.0, 1.0)

            # 以训练默认位姿为中心对齐：obs = default + gain * (eq - neutral)
            tendon = self.default_tendon6 + self.obs_align_gain * (eq_tendon - neutral_tendon)
            tendon = np.clip(tendon, 0.0, 1.0).astype(np.float32)

            # thumb_abd 用拇指旋转通道做同样中心对齐。
            eq_thumb_rot = float(np.clip(eq_cmd_deg[0] / 90.0, 0.0, 1.0))
            neutral_thumb_rot = float(np.clip(neutral_eq_deg[0] / 90.0, 0.0, 1.0))
            thumb_abd = self.default_thumb_abd + self.obs_align_gain * (eq_thumb_rot - neutral_thumb_rot)
            thumb_abd = float(np.clip(thumb_abd, 0.0, 1.0))

            return tendon, thumb_abd

        tendon = np.asarray(fb.get("tendon_lengths", np.zeros(6)), dtype=np.float32).reshape(6)
        thumb_abd = float(fb.get("thumb_abd", tendon[4]))
        return tendon, thumb_abd

    def _build_policy_obs(self) -> dict:
        # 构造策略输入：
        # state_t = [6腱长, 1拇指外展, 7维last_action] 共14维。
        # 然后做 history 堆叠，保持和仿真部署脚本一致。
        tendon6, thumb_abd = self._read_feedback_state()
        self.last_obs_tendon6 = np.asarray(tendon6, dtype=np.float32).copy()
        self.last_obs_thumb_abd = float(thumb_abd)
        state = jp.array(np.concatenate([tendon6, np.array([thumb_abd], dtype=np.float32)]), dtype=jp.float32)
        state = jp.concatenate([state, self.last_action])

        self.obs_history = jp.roll(self.obs_history, state.size)
        self.obs_history = self.obs_history.at[: state.size].set(state)

        return {"state": np.asarray(self.obs_history, dtype=np.float32)}

    def _targets7_to_targets6(self, targets7: np.ndarray) -> np.ndarray:
        # 7->6 映射：
        # - 四指保持直连：[0,1,2,3]
        # - 拇指两路由策略的3路拇指相关目标融合得到，避免直接丢掉第4维。
        t7 = np.asarray(targets7, dtype=np.float32).reshape(7)
        t6 = np.zeros(6, dtype=np.float32)
        t6[:4] = t7[:4]
        t6[4] = self.thumb_flex_w4 * t7[4] + self.thumb_flex_w5 * t7[5]
        t6[5] = self.thumb_rot_w4 * t7[4] + self.thumb_rot_w6 * t7[6]
        return np.clip(t6, 0.0, 1.0)

    def _apply_hw6_cmd_calibration(self, desired6: np.ndarray) -> np.ndarray:
        # desired6 语义：期望回读（归一化 0~1）。
        # 用线性反解得到更贴近目标回读的下发值。
        desired_deg = np.clip(np.asarray(desired6, dtype=np.float32), 0.0, 1.0) * 90.0
        safe_k = np.where(np.abs(self.hw6_k) < 1e-6, 1.0, self.hw6_k)
        cmd_deg = (desired_deg - self.hw6_b) / safe_k
        cmd_norm = np.clip(cmd_deg / 90.0, 0.0, 1.0)
        return cmd_norm.astype(np.float32)

    def _rate_limit_targets(self, targets7: np.ndarray) -> np.ndarray:
        # 目标限速：每步变化不超过 max_target_delta。
        delta = np.clip(targets7 - self.last_targets7, -self.max_target_delta, self.max_target_delta)
        return self.last_targets7 + delta

    def _debug_targets7(self) -> np.ndarray:
        # 生成小幅周期目标用于链路联调，不依赖策略与真实反馈。
        t = self._step_idx * self.control_dt
        phase = (2.0 * np.pi * t) / self.debug_period_s
        # default_tendon 来自 jax 数组，这里显式 copy 成可写 numpy 数组。
        targets7 = np.array(self.default_tendon, dtype=np.float32, copy=True)
        # 仅在 6 个可映射通道做小幅扰动，避免大动作冲击。
        debug_idx = [0, 1, 2, 3, 5, 6]
        for i, idx in enumerate(debug_idx):
            wave = np.sin(phase + i * (np.pi / 3.0))
            targets7[idx] = float(self.default_tendon[idx]) + self.debug_amp_norm * wave
        return np.clip(targets7, 0.0, 1.0)

    def step_once(self) -> None:
        # 闭环一步：读反馈 -> 推理 -> 动作映射 -> 限幅限速 -> 下发。
        if self.control_mode == "rl":
            obs = self._build_policy_obs()
            self.rng, act_rng = jax.random.split(self.rng)
            action_jax, _ = self.policy_fn(obs, act_rng)
            action = np.asarray(action_jax, dtype=np.float32).reshape(7)

            # 与仿真策略一致的动作到目标映射。
            action_scaled = action * self.action_gain
            targets7 = np.asarray(self.default_tendon + action_scaled * self.action_scale, dtype=np.float32)
            targets7 = np.clip(targets7, 0.0, 1.0)
            self.last_action = jp.array(action_scaled, dtype=jp.float32)
            self.last_action_np = action_scaled.copy()
            self.last_raw_action_np = action.copy()
        elif self.control_mode in ("observe", "observe_relaxed"):
            # 只更新观测链路，不发送任何控制命令。
            _ = self._build_policy_obs()
            self.last_action = jp.zeros((7,), dtype=jp.float32)
            self.last_action_np = np.zeros((7,), dtype=np.float32)
            self.last_raw_action_np = np.zeros((7,), dtype=np.float32)
            self._step_idx += 1
            return
        else:
            targets7 = self._debug_targets7()
            self.last_action = jp.zeros((7,), dtype=jp.float32)
            self.last_action_np = np.zeros((7,), dtype=np.float32)
            self.last_raw_action_np = np.zeros((7,), dtype=np.float32)

        targets7 = self._rate_limit_targets(targets7)

        targets6 = self._targets7_to_targets6(targets7)
        targets6 = self._apply_hw6_cmd_calibration(targets6)
        self.hand.send_targets6(targets6)

        self.last_targets7 = targets7
        self._step_idx += 1

    def _write_csv_row(self, step_count: int) -> None:
        if self._csv_writer is None:
            return

        t6 = self._targets7_to_targets6(self.last_targets7)
        fb_dbg = self.hand.feedback_debug()
        qangles = fb_dbg["last_query_angles_deg"]
        q = np.full(6, np.nan, dtype=np.float32) if qangles is None else np.asarray(qangles, dtype=np.float32).reshape(6)

        if self._baseline_q_angles is None and qangles is not None and step_count >= self._baseline_warmup_steps:
            self._baseline_q_angles = q.copy()
            self._baseline_set_step = int(step_count)
            if self.verbose:
                log(f"baseline set at step={self._baseline_set_step}: {np.round(self._baseline_q_angles, 2).tolist()}")

        if self._baseline_q_angles is None:
            dq = np.full(6, np.nan, dtype=np.float32)
        else:
            dq = q - self._baseline_q_angles

        row = {
            "time": time.time(),
            "step": int(step_count),
            "mode": self.control_mode,
            "target6_0": float(t6[0]), "target6_1": float(t6[1]), "target6_2": float(t6[2]),
            "target6_3": float(t6[3]), "target6_4": float(t6[4]), "target6_5": float(t6[5]),
            "target7_0": float(self.last_targets7[0]), "target7_1": float(self.last_targets7[1]), "target7_2": float(self.last_targets7[2]),
            "target7_3": float(self.last_targets7[3]), "target7_4": float(self.last_targets7[4]), "target7_5": float(self.last_targets7[5]),
            "target7_6": float(self.last_targets7[6]),
            "action_norm": float(np.linalg.norm(self.last_action_np)),
            "raw_action_0": float(self.last_raw_action_np[0]), "raw_action_1": float(self.last_raw_action_np[1]), "raw_action_2": float(self.last_raw_action_np[2]),
            "raw_action_3": float(self.last_raw_action_np[3]), "raw_action_4": float(self.last_raw_action_np[4]), "raw_action_5": float(self.last_raw_action_np[5]),
            "raw_action_6": float(self.last_raw_action_np[6]),
            "obs_tendon6_0": float(self.last_obs_tendon6[0]), "obs_tendon6_1": float(self.last_obs_tendon6[1]),
            "obs_tendon6_2": float(self.last_obs_tendon6[2]), "obs_tendon6_3": float(self.last_obs_tendon6[3]),
            "obs_tendon6_4": float(self.last_obs_tendon6[4]), "obs_tendon6_5": float(self.last_obs_tendon6[5]),
            "obs_thumb_abd": float(self.last_obs_thumb_abd),
            "fb_source": fb_dbg["source"],
            "q_ok": int(fb_dbg["query_ok"]),
            "q_fail": int(fb_dbg["query_fail"]),
            "q_angle_deg_0": float(q[0]), "q_angle_deg_1": float(q[1]), "q_angle_deg_2": float(q[2]),
            "q_angle_deg_3": float(q[3]), "q_angle_deg_4": float(q[4]), "q_angle_deg_5": float(q[5]),
            "dq_angle_deg_0": float(dq[0]), "dq_angle_deg_1": float(dq[1]), "dq_angle_deg_2": float(dq[2]),
            "dq_angle_deg_3": float(dq[3]), "dq_angle_deg_4": float(dq[4]), "dq_angle_deg_5": float(dq[5]),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def run(self) -> None:
        next_t = time.perf_counter()
        step_count = 0
        try:
            if self.control_mode == "rl" and self.rl_preopen_s > 0.0:
                # RL 启动前先把手带到开手位，减少与训练初态偏差。
                preopen_n = max(1, int(round(self.rl_preopen_s / max(self.control_dt, 1e-3))))
                for _ in range(preopen_n):
                    self.hand.send_targets6(np.zeros(6, dtype=np.float32))
                    time.sleep(self.control_dt)
                if self.verbose:
                    log(f"rl startup preopen done: {self.rl_preopen_s:.2f}s")

            if self.control_mode == "observe_relaxed":
                # 进入只观测+放松模式：先主动开手，再切换到放松观测。
                self.hand.set_force_mode(False)
                time.sleep(0.05)

                # 先主动发一段时间开手目标，尽量把手带到伸直位。
                preopen_n = max(1, int(round(self.observe_preopen_s / max(self.control_dt, 1e-3))))
                for _ in range(preopen_n):
                    self.hand.send_targets6(np.zeros(6, dtype=np.float32))
                    time.sleep(self.control_dt)

                # 再切到全通道 disable，尽量减少位置保持影响。
                self.hand.send_relax6()
                time.sleep(0.05)
                if self.verbose:
                    log(
                        "observe_relaxed startup: force_mode=off + "
                        f"preopen={self.observe_preopen_s:.2f}s + relax6 sent"
                    )

            if self.verbose:
                log(f"control loop started (dt={self.control_dt:.3f}s)")
            while True:
                self.step_once()
                step_count += 1
                self._write_csv_row(step_count)
                if self.verbose and step_count % self.heartbeat_steps == 0:
                    t6 = self._targets7_to_targets6(self.last_targets7)
                    fb_dbg = self.hand.feedback_debug()
                    action_norm = float(np.linalg.norm(self.last_action_np))
                    src = fb_dbg["source"]
                    qok = fb_dbg["query_ok"]
                    qfail = fb_dbg["query_fail"]
                    qangles = fb_dbg["last_query_angles_deg"]
                    qangles_txt = "none" if qangles is None else np.round(qangles, 1).tolist()
                    if self._baseline_q_angles is not None and qangles is not None:
                        dq = np.asarray(qangles, dtype=np.float32) - self._baseline_q_angles
                        dq_txt = np.round(dq, 2).tolist()
                    else:
                        dq_txt = "none"
                    log(
                        f"alive step={step_count} targets6={np.round(t6, 3).tolist()} "
                        f"action_norm={action_norm:.4f} fb={src} q_ok={qok} q_fail={qfail} "
                        f"q_angles_deg={qangles_txt} dq_angles_deg={dq_txt}"
                    )
                next_t += self.control_dt
                sleep_t = next_t - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.perf_counter()

                if self.max_steps > 0 and step_count >= self.max_steps:
                    if self.verbose:
                        log(f"max_steps reached ({self.max_steps}), stop loop")
                    break
        finally:
            self.hand.close()
            if self._csv_file is not None:
                self._csv_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Aero RL policy to hardware serial bridge")
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH, help="Path to Orbax checkpoint directory")
    parser.add_argument("--serial_port", default=DEFAULT_SERIAL_PORT, help="Hardware serial port")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baudrate")
    parser.add_argument("--env_name", default=DEFAULT_ENV_NAME, help="Policy environment name")
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN, help="Observation history length")
    parser.add_argument("--control_dt", type=float, default=DEFAULT_CONTROL_DT, help="Control period in seconds")
    parser.add_argument("--max_target_delta", type=float, default=DEFAULT_MAX_TARGET_DELTA, help="Max per-step target delta")
    parser.add_argument("--quiet", action="store_true", help="Disable startup/heartbeat logs")
    parser.add_argument("--heartbeat_steps", type=int, default=20, help="Print heartbeat every N control steps")
    parser.add_argument("--mode", choices=["rl", "debug", "observe", "observe_relaxed"], default="debug", help="Control mode")
    parser.add_argument("--debug_amp_deg", type=float, default=8.0, help="Debug mode wave amplitude in degrees")
    parser.add_argument("--debug_period_s", type=float, default=2.5, help="Debug mode wave period in seconds")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop automatically after N steps, 0 means run forever")
    parser.add_argument("--log_csv", default=DEFAULT_LOG_CSV, help="CSV path for per-step runtime logs, empty to disable")
    parser.add_argument(
        "--observe_preopen_s",
        type=float,
        default=DEFAULT_OBSERVE_PREOPEN_S,
        help="observe_relaxed startup pre-open duration in seconds",
    )
    parser.add_argument(
        "--rl_preopen_s",
        type=float,
        default=DEFAULT_RL_PREOPEN_S,
        help="rl mode startup pre-open duration in seconds",
    )
    parser.add_argument(
        "--action_gain",
        type=float,
        default=DEFAULT_ACTION_GAIN,
        help="global gain on RL action before mapping, range [0,1]",
    )
    parser.add_argument(
        "--thumb_flex_w4",
        type=float,
        default=DEFAULT_THUMB_FLEX_W4,
        help="thumb flex cmd from target7[4] weight",
    )
    parser.add_argument(
        "--thumb_flex_w5",
        type=float,
        default=DEFAULT_THUMB_FLEX_W5,
        help="thumb flex cmd from target7[5] weight",
    )
    parser.add_argument(
        "--thumb_rot_w4",
        type=float,
        default=DEFAULT_THUMB_ROT_W4,
        help="thumb rot cmd from target7[4] weight",
    )
    parser.add_argument(
        "--thumb_rot_w6",
        type=float,
        default=DEFAULT_THUMB_ROT_W6,
        help="thumb rot cmd from target7[6] weight",
    )
    # 常用调参建议：
    # - 控制不稳/抖动：减小 --max_target_delta。
    # - 响应太慢：增大 --max_target_delta（需注意安全）。
    # - 时序不匹配：优先保持 --control_dt 与训练控制周期一致。
    return parser.parse_args()


def main():
    args = parse_args()

    log("script start")
    log(f"mode={args.mode}")
    log(f"checkpoint={args.checkpoint_path}")
    log(f"serial={args.serial_port} baud={args.baudrate}")

    if args.mode == "rl":
        ckpt_path = epath.Path(args.checkpoint_path)
        if "REPLACE_WITH_YOUR_CHECKPOINT_DIR" in str(ckpt_path) or not ckpt_path.exists():
            raise FileNotFoundError(
                "checkpoint_path 无效。请在脚本顶部修改 DEFAULT_CHECKPOINT_PATH，"
                "指向你的 Orbax checkpoint 目录，然后直接在 VS Code 运行。"
            )

    bridge = AeroHardwareBridge(
        checkpoint_path=args.checkpoint_path,
        serial_port=args.serial_port,
        serial_baudrate=args.baudrate,
        env_name=args.env_name,
        history_len=args.history_len,
        control_dt=args.control_dt,
        max_target_delta=args.max_target_delta,
        verbose=not args.quiet,
        heartbeat_steps=args.heartbeat_steps,
        control_mode=args.mode,
        debug_amp_deg=args.debug_amp_deg,
        debug_period_s=args.debug_period_s,
        max_steps=args.max_steps,
        log_csv_path=(args.log_csv.strip() if args.log_csv is not None else ""),
        observe_preopen_s=args.observe_preopen_s,
        rl_preopen_s=args.rl_preopen_s,
        action_gain=args.action_gain,
        thumb_flex_w4=args.thumb_flex_w4,
        thumb_flex_w5=args.thumb_flex_w5,
        thumb_rot_w4=args.thumb_rot_w4,
        thumb_rot_w6=args.thumb_rot_w6,
    )

    bridge.run()


if __name__ == "__main__":
    main()

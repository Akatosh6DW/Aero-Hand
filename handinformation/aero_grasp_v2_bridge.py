"""AeroCubeGraspV2ForceCoacd 策略 -> 实物手串口桥接脚本。

这版桥接以 `aero_grasp_hw6force_bridge.py` 为底稿，但面向当前 V2/Coacd
checkpoint，重点补了三件事：

1. 正确恢复新 checkpoint：
   - 合并 checkpoint 自带 `config.json`
   - 合并 checkpoint 自带 `ppo_network_config.json`
   - 不再把策略输入硬编码成旧版 17 维
2. 接入 V2 的 46 维状态：
   - 真正把指尖触觉/力反馈喂回策略
   - 支持 phase / hold progress / force balance 一并构造
3. 从 URDF 目录读取实物手元数据：
   - 右手关节名、关节链、指序
   - 方便后续把桥接日志和实物机械结构对齐

重要说明：
- V2 state 里与方块姿态/位置强相关的 25 维观测
  (`cube_pos_error/cube_vel/cube_quat/fingertip_to_cube`) 在纯串口硬件链路里
  没有直接传感器来源。本脚本默认用“安全中性占位”：
  - cube_pos_error = 0
  - cube_vel = 0
  - cube_quat = [1, 0, 0, 0]
  - fingertip_to_cube = 0
- 这意味着它在逻辑上能接当前 checkpoint，但如果要完全发挥 V2 策略，
  仍建议后续接入外部物体位姿估计。

使用方式：
  conda activate aero_rl
  python aero_grasp_v2_bridge.py --dry_run_policy
  python aero_grasp_v2_bridge.py --mode rl
  python aero_grasp_v2_bridge.py --mode observe
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jp
import ml_collections
import numpy as np
import serial
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train
from etils import epath
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params
from mujoco_playground._src import manipulation


# ──────────────────── 默认配置（VS Code 一键运行） ──────────────────── #
DEFAULT_CHECKPOINT_PATH = (
    "/root/autodl-tmp/Aero-Hand/sim_rl/mujoco_playground/logs/"
    "AeroCubeGraspV2ForceCoacd-20260421-035739-C30_30s_push_24576/"
    "checkpoints/000011796480"
)
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 115200
DEFAULT_ENV_NAME = "AeroCubeGraspV2ForceCoacd"
DEFAULT_CONTROL_DT = 0.05
DEFAULT_MAX_TARGET_DELTA = 0.03
DEFAULT_LOG_CSV = "/root/autodl-tmp/Aero-Hand/handinformation/aero_bridge_v2_runlog.csv"
DEFAULT_PREOPEN_S = 1.2
DEFAULT_ACTION_GAIN = 1.0
DEFAULT_FORCE_SAFETY_THRESHOLD = 0.0

REPO_ROOT = Path("/root/autodl-tmp/Aero-Hand")
URDF_ROOT = REPO_ROOT / "handinformation" / "URDF" / "右" / "qbr"
QBR_URDF_PATH = URDF_ROOT / "urdf" / "qbr.urdf"
QBR_CSV_PATH = URDF_ROOT / "urdf" / "qbr.csv"
QBR_JOINT_YAML_PATH = URDF_ROOT / "config" / "joint_names_qbr.yaml"

# 训练里的右手 MJCF，里面有 actuator / tactile site 的精确定义。
V2_MJCF_PATH = (
    REPO_ROOT
    / "sim_rl"
    / "mujoco_playground"
    / "mujoco_playground"
    / "_src"
    / "manipulation"
    / "aero_hand"
    / "xmls"
    / "right_hand_v2_vertical_coacd.xml"
)

# 训练 ctrl 空间单位：[thumb_rot(rad), thumb_flex(m), index(m), middle(m), ring(m), pinky(m)]
CTRL_RANGES = np.array(
    [
        [-0.1000, 1.7500],
        [0.0816, 0.1121],
        [0.0585, 0.1104],
        [0.0585, 0.1104],
        [0.0585, 0.1104],
        [0.0585, 0.1104],
    ],
    dtype=np.float32,
)
DEFAULT_TENDON = np.array(
    [0.0, 0.1099, 0.1074, 0.1081, 0.1074, 0.1093], dtype=np.float32
)
ACTION_SCALE = np.array(
    [1.3, 0.03, 0.055, 0.055, 0.055, 0.055], dtype=np.float32
)

# readback_deg ≈ k * cmd_deg + b
HW6_K = np.array(
    [0.9933, 0.9928, 1.0018, 1.0029, 1.0004, 1.0002], dtype=np.float32
)
HW6_B = np.array(
    [-0.2371, -0.5978, -0.2071, -0.2094, -0.2409, -0.0640], dtype=np.float32
)

TAXEL_WEIGHTS = np.array(
    [
        0.7, 1.0, 1.0, 0.7,
        1.0, 1.4, 1.4, 1.0,
        1.0, 1.4, 1.4, 1.0,
        0.7, 1.0, 1.0, 0.7,
    ],
    dtype=np.float32,
)

# 触觉串口原始顺序：[thumb, index, middle, ring, pinky]
# V2 env tactile 顺序：tip_force = [index, middle, ring, pinky, thumb]
TACTILE_HW_TO_ENV = np.array([1, 2, 3, 4, 0], dtype=np.int32)
TACTILE_ENV_TO_HW = np.array([4, 0, 1, 2, 3], dtype=np.int32)


def log(msg: str) -> None:
    print(f"[v2_bridge] {msg}", flush=True)


def ctrl_to_deg(ctrl6: np.ndarray) -> np.ndarray:
    deg = np.zeros(6, dtype=np.float32)
    deg[0] = np.clip(ctrl6[0], 0.0, None) * (180.0 / np.pi)
    for i in range(1, 6):
        lo, hi = CTRL_RANGES[i]
        rng = hi - lo
        deg[i] = (hi - ctrl6[i]) / rng * 90.0 if rng > 1e-8 else 0.0
    return np.clip(deg, 0.0, 90.0)


def deg_to_ctrl(deg6: np.ndarray) -> np.ndarray:
    ctrl = np.zeros(6, dtype=np.float32)
    ctrl[0] = np.clip(deg6[0], 0.0, 90.0) * (np.pi / 180.0)
    for i in range(1, 6):
        lo, hi = CTRL_RANGES[i]
        rng = hi - lo
        ctrl[i] = hi - (np.clip(deg6[i], 0.0, 90.0) / 90.0) * rng
    return ctrl


def _deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_config_dict(
    base: ml_collections.ConfigDict, overlay: dict[str, Any]
) -> ml_collections.ConfigDict:
    for key, value in overlay.items():
        if isinstance(value, dict):
            if key not in base or not isinstance(base[key], ml_collections.ConfigDict):
                base[key] = ml_collections.ConfigDict()
            _merge_config_dict(base[key], value)
        else:
            base[key] = value
    return base


def _as_plain_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)


def _resolve_checkpoint_step_dir(checkpoint_path: str | Path) -> Path:
    p = Path(checkpoint_path).expanduser().resolve()
    if p.is_file():
        raise ValueError(f"checkpoint_path 应该指向目录，不是文件: {p}")
    if (p / "ppo_network_config.json").exists():
        return p
    if (p / "checkpoints").is_dir():
        p = p / "checkpoints"
    numbered = sorted(d for d in p.iterdir() if d.is_dir() and d.name.isdigit())
    if numbered:
        return numbered[-1]
    raise FileNotFoundError(f"未在 {checkpoint_path} 下找到有效 checkpoint step 目录。")


def _checkpoint_root_from_step(step_dir: Path) -> Path:
    return step_dir.parent


def _load_checkpoint_json(step_dir: Path, name: str) -> dict[str, Any]:
    if name == "ppo_network_config.json":
        path = step_dir / name
    else:
        path = _checkpoint_root_from_step(step_dir) / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_env_and_network(env_name: str, checkpoint_path: str):
    step_dir = _resolve_checkpoint_step_dir(checkpoint_path)
    env_cfg = manipulation.get_default_config(env_name)
    ckpt_cfg = _load_checkpoint_json(step_dir, "config.json")
    if ckpt_cfg:
        _merge_config_dict(env_cfg, ckpt_cfg)

    ppo_params = manipulation_params.brax_ppo_config(env_name)
    network_factory_config = dict(ppo_params.get("network_factory", {}))
    ckpt_network_cfg = _load_checkpoint_json(step_dir, "ppo_network_config.json")
    if ckpt_network_cfg:
        allowed_keys = {
            "policy_hidden_layer_sizes",
            "value_hidden_layer_sizes",
            "policy_obs_key",
            "value_obs_key",
            "distribution_type",
            "noise_std_type",
            "init_noise_std",
            "state_dependent_std",
            "mean_clip_scale",
            "use_distributional_critic",
            "num_quantiles",
        }
        for key, value in ckpt_network_cfg.get("network_factory_kwargs", {}).items():
            if key in allowed_keys:
                network_factory_config[key] = value
        if "normalize_observations" in ckpt_network_cfg:
            ppo_params["normalize_observations"] = ckpt_network_cfg["normalize_observations"]

    for key in ("num_timesteps", "network_factory"):
        if key in ppo_params:
            del ppo_params[key]

    env = registry.load(env_name, config=env_cfg)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **network_factory_config,
    )
    make_inference_fn, params, _ = ppo_train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        num_timesteps=0,
        seed=1,
        restore_checkpoint_path=epath.Path(step_dir),
        **ppo_params,
    )

    if isinstance(params, dict):
        normalizer = params.get("normalizer") or params.get("normalizer_params")
        policy = params.get("policy") or params.get("policy_params") or params.get("params")
        if normalizer is not None and policy is not None:
            params = (normalizer, policy)

    obs_size = ckpt_network_cfg.get("observation_size", {}).get("state", {}).get("shape", [46])[0]
    jit_inference = jax.jit(make_inference_fn(params, deterministic=True))

    return {
        "policy_fn": jit_inference,
        "step_dir": step_dir,
        "env_cfg": env_cfg,
        "env_cfg_dict": _as_plain_dict(env_cfg),
        "network_factory_config": network_factory_config,
        "obs_size": int(obs_size),
        "action_size": int(ckpt_network_cfg.get("action_size", env.action_size)),
    }


@dataclass(frozen=True)
class FingerChain:
    name: str
    joints: tuple[str, ...]
    fingertip_link: str
    lateral_x: float


@dataclass(frozen=True)
class V2HandMetadata:
    controller_joint_names: tuple[str, ...]
    thumb_chain: FingerChain
    fingers: tuple[FingerChain, ...]
    actuator_joint_map: tuple[tuple[str, str], ...]
    tactile_site_map: tuple[tuple[str, tuple[str, ...]], ...]


def _parse_joint_names_yaml(yaml_path: Path) -> tuple[str, ...]:
    text = yaml_path.read_text(encoding="utf-8")
    if "[" in text and "]" in text:
        inner = text.split("[", 1)[1].rsplit("]", 1)[0]
        names = [item.strip().strip("'\"") for item in inner.split(",")]
        return tuple(name for name in names if name)
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("-"):
            name = line[1:].strip().strip("'\"")
            if name:
                names.append(name)
    return tuple(names)


def _load_qbr_hand_metadata() -> V2HandMetadata:
    controller_joint_names = _parse_joint_names_yaml(QBR_JOINT_YAML_PATH)
    rows: list[dict[str, str]] = []
    with QBR_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    rows_by_joint = {row["Joint Name"]: row for row in rows if row.get("Joint Name")}
    child_to_joint: dict[str, str] = {}
    for row in rows:
        joint_name = row.get("Joint Name", "")
        child_link = row.get("Link Name", "")
        if joint_name and child_link:
            child_to_joint[child_link] = joint_name

    def _origin_x(joint_name: str) -> float:
        return float(rows_by_joint[joint_name]["Joint Origin X"])

    thumb_chain = FingerChain(
        name="thumb",
        joints=("j1", "j2", "j3"),
        fingertip_link="link3",
        lateral_x=_origin_x("j1"),
    )

    non_thumb_bases = ["j4", "j6", "j8", "j10"]
    sorted_bases = sorted(non_thumb_bases, key=_origin_x, reverse=True)
    finger_names = ("index", "middle", "ring", "pinky")
    fingers: list[FingerChain] = []
    for name, base_joint in zip(finger_names, sorted_bases):
        base_row = rows_by_joint[base_joint]
        proximal_link = base_row["Link Name"]
        distal_link = ""
        distal_joint = ""
        for row in rows:
            if row.get("Parent") == proximal_link and row.get("Joint Name"):
                distal_joint = row["Joint Name"]
                distal_link = row["Link Name"]
                break
        if not distal_joint or not distal_link:
            raise RuntimeError(f"无法从 URDF CSV 推断 {name} 的 distal joint/link。")
        fingers.append(
            FingerChain(
                name=name,
                joints=(base_joint, distal_joint),
                fingertip_link=distal_link,
                lateral_x=_origin_x(base_joint),
            )
        )

    mjcf_root = ET.parse(V2_MJCF_PATH).getroot()
    actuator_joint_map: list[tuple[str, str]] = []
    actuator_elem = mjcf_root.find("actuator")
    if actuator_elem is not None:
        for child in actuator_elem:
            actuator_name = child.attrib.get("name")
            joint_name = child.attrib.get("joint")
            if actuator_name and joint_name:
                actuator_joint_map.append((actuator_name, joint_name))

    tactile_sites: dict[str, list[str]] = {
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": [],
    }
    sensor_elem = mjcf_root.find("sensor")
    if sensor_elem is not None:
        for child in sensor_elem:
            name = child.attrib.get("name", "")
            site = child.attrib.get("site", "")
            if not name.startswith("hw_tip_frc_") or not site:
                continue
            parts = name.split("_")
            finger = parts[3]
            if finger in tactile_sites:
                tactile_sites[finger].append(site)
    tactile_site_map = tuple(
        (name, tuple(sorted(sites))) for name, sites in tactile_sites.items()
    )

    return V2HandMetadata(
        controller_joint_names=controller_joint_names,
        thumb_chain=thumb_chain,
        fingers=tuple(fingers),
        actuator_joint_map=tuple(actuator_joint_map),
        tactile_site_map=tactile_site_map,
    )


class HandSerialProtocol:
    """串口协议封装（动作控制 + 角度查询 + 触觉读取）。"""

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

        self._tactile_raw = np.zeros((5, 16), dtype=np.uint16)
        self._tactile_ts = np.zeros(5, dtype=np.float64)
        self._tactile_lock = threading.Lock()
        self._tactile_update_count = 0

        self._angle_reply: Optional[np.ndarray] = None
        self._angle_event = threading.Event()

        self._rx_running = True
        self._rx_buf = bytearray()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, daemon=True, name="serial_rx"
        )
        self._rx_thread.start()

    def _rx_loop(self) -> None:
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
        buf = self._rx_buf
        while len(buf) >= 7:
            if buf[0] != self.START:
                buf.pop(0)
                continue

            if len(buf) < 4:
                break

            if buf[1] in (1, 2, 3, 4, 5) and buf[2] == 0x40 and buf[3] == 0x83:
                if len(buf) < self.TACTILE_FRAME_LEN:
                    break
                frame = bytes(buf[: self.TACTILE_FRAME_LEN])
                del buf[: self.TACTILE_FRAME_LEN]
                self._handle_tactile_frame(frame)
                continue

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

            if buf[1] in (0x40, 0x4A, 0x4B):
                flen = 12 if buf[1] == 0x4B else 7
                if len(buf) < flen:
                    break
                del buf[:flen]
                continue

            buf.pop(0)

    def _handle_tactile_frame(self, frame: bytes) -> None:
        if frame[63] != self.END:
            return
        finger_id = frame[1]
        if finger_id < 1 or finger_id > 5:
            return
        cksum = sum(frame[3:62]) & 0xFF
        if cksum != frame[62]:
            return
        taxels = np.zeros(16, dtype=np.uint16)
        for i in range(16):
            off = 4 + 2 * i
            taxels[i] = (frame[off] << 8) | frame[off + 1]
        idx = finger_id - 1
        with self._tactile_lock:
            self._tactile_raw[idx] = taxels
            self._tactile_ts[idx] = time.time()
            self._tactile_update_count += 1

    def get_tactile_raw(self) -> np.ndarray:
        with self._tactile_lock:
            return self._tactile_raw.copy()

    def get_tactile_pooled_hw(self) -> np.ndarray:
        with self._tactile_lock:
            raw = self._tactile_raw.astype(np.float32)
        return np.sum(raw * TAXEL_WEIGHTS[None, :], axis=1)

    def get_tactile_pooled_env(self) -> np.ndarray:
        hw = self.get_tactile_pooled_hw()
        return hw[TACTILE_HW_TO_ENV]

    @staticmethod
    def _checksum(body: list[int]) -> int:
        return sum(body) & 0xFF

    def _send_body(self, body: list[int]) -> None:
        packet = [self.START] + body + [self._checksum(body), self.END]
        self.ser.write(bytes(packet))

    @staticmethod
    def _deg_to_u8(angle_deg: float) -> int:
        a = float(np.clip(angle_deg, 0.0, 90.0))
        return int(round(a))

    def send_targets_deg(self, targets_deg6: np.ndarray) -> None:
        self.send_targets_deg_with_enable(targets_deg6, np.ones(6, dtype=np.int32))

    def send_targets_deg_with_enable(
        self, targets_deg6: np.ndarray, enables6: np.ndarray
    ) -> None:
        targets_deg6 = np.asarray(targets_deg6, dtype=np.float32).reshape(6)
        enables6 = np.asarray(enables6, dtype=np.int32).reshape(6)
        enables6 = np.where(enables6 > 0, 1, 0)

        body = [0x10, 0x11]
        for i in range(6):
            body.append(int(enables6[i]))
            body.append(self._deg_to_u8(float(targets_deg6[i])))
        self._send_body(body)

    def send_open(self) -> None:
        self.send_targets_deg(np.zeros(6, dtype=np.float32))

    def send_relax(self) -> None:
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

    def _send_query_angle(self) -> None:
        body = [self.CMD_QUERY_ANGLE, self.LEN_QUERY, 0x00, 0x00]
        self._send_body(body)

    @staticmethod
    def _parse_query_angle_reply(frame12: bytes) -> Optional[np.ndarray]:
        if len(frame12) != 12:
            return None
        if frame12[0] != 0x5A or frame12[1] != 0xF1 or frame12[11] != 0x5D:
            return None
        d1_to_d9 = list(frame12[1:10])
        if (sum(d1_to_d9) & 0xFF) != int(frame12[10]):
            return None
        return np.frombuffer(bytes(frame12[4:10]), dtype=np.uint8).astype(np.float32)

    def query_angles_once(self) -> Optional[np.ndarray]:
        self._angle_event.clear()
        self._send_query_angle()
        if self._angle_event.wait(timeout=0.05):
            return self._angle_reply
        self._query_fail += 1
        return None

    def feedback_debug(self) -> dict[str, Any]:
        with self._tactile_lock:
            t_count = self._tactile_update_count
            t_ts = self._tactile_ts.copy()
        return {
            "source": "query" if self._last_query_angles_deg is not None else "none",
            "query_ok": int(self._query_ok),
            "query_fail": int(self._query_fail),
            "last_query_angles_deg": (
                None if self._last_query_angles_deg is None else self._last_query_angles_deg.copy()
            ),
            "tactile_updates": int(t_count),
            "tactile_age_s": (
                round(time.time() - float(np.max(t_ts)), 3) if np.any(t_ts > 0) else -1.0
            ),
        }

    def close(self) -> None:
        self._rx_running = False
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        if self.ser.is_open:
            self.ser.close()


class V2ObservationProjector:
    """把硬件可见量投影到 V2 策略需要的 46D state。"""

    def __init__(self, tactile_saturation_n: float, hold_success_sec: float = 30.0):
        self.tactile_saturation_n = float(max(1e-6, tactile_saturation_n))
        self.hold_success_sec = float(max(1e-6, hold_success_sec))
        self.start_time = time.time()

    def build_state(
        self,
        motor_targets: np.ndarray,
        tactile_pooled_env: np.ndarray,
        last_action: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        tactile_norm = np.clip(tactile_pooled_env / self.tactile_saturation_n, 0.0, 1.0)
        # 与 V2 env 对齐：primary = index(0), middle(1), thumb(4)
        primary_forces = np.abs(
            np.asarray([tactile_pooled_env[0], tactile_pooled_env[1], tactile_pooled_env[4]])
        )
        mean_force = float(np.mean(primary_forces))
        rel_std = float(np.std(primary_forces) / (mean_force + 1e-6))
        force_balance_obs = float(
            np.clip(1.0 - rel_std, 0.0, 1.0) * np.clip(mean_force / 0.1, 0.0, 1.0)
        )

        hold_duration_normalized = float(
            np.clip((time.time() - self.start_time) / self.hold_success_sec, 0.0, 1.0)
        )

        cube_pos_error = np.zeros(3, dtype=np.float32)
        cube_vel_scaled = np.zeros(3, dtype=np.float32)
        cube_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        fingertip_to_cube_scaled = np.zeros(15, dtype=np.float32)
        support_phase = np.array([1.0, 1.0], dtype=np.float32)
        tail = np.array([hold_duration_normalized, force_balance_obs], dtype=np.float32)

        state = np.concatenate(
            [
                motor_targets.astype(np.float32),
                tactile_norm.astype(np.float32),
                last_action.astype(np.float32),
                cube_pos_error,
                cube_vel_scaled,
                cube_quat,
                fingertip_to_cube_scaled,
                support_phase,
                tail,
            ]
        ).astype(np.float32)

        aux = {
            "force_balance_obs": force_balance_obs,
            "hold_duration_normalized": hold_duration_normalized,
            "support_pre": 1.0,
            "support_post": 1.0,
        }
        return state, aux


def load_policy_bundle(env_name: str, checkpoint_path: str):
    bundle = _load_env_and_network(env_name, checkpoint_path)
    log("JIT warmup ...")
    dummy_obs = {"state": np.zeros(bundle["obs_size"], dtype=np.float32)}
    rng = jax.random.PRNGKey(0)
    _ = bundle["policy_fn"](dummy_obs, rng)
    log("JIT warmup done")
    return bundle


class AeroV2Bridge:
    """V2/Coacd 策略到硬件的实时桥接。"""

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
        self.hand_metadata = _load_qbr_hand_metadata()

        if self.verbose:
            log(f"loading V2 policy: {checkpoint_path}")
        self.policy_bundle = load_policy_bundle(env_name, checkpoint_path)
        self.policy_fn = self.policy_bundle["policy_fn"]
        self.obs_projector = V2ObservationProjector(
            tactile_saturation_n=self.policy_bundle["env_cfg_dict"]["tactile_config"]["force_saturation_n"],
            hold_success_sec=self.policy_bundle["env_cfg_dict"]["reward_config"].get("hold_success_sec", 30.0),
        )
        self.obs_size = int(self.policy_bundle["obs_size"])
        self.action_size = int(self.policy_bundle["action_size"])
        if self.obs_size != 46 or self.action_size != 6:
            raise RuntimeError(
                f"当前桥接只支持 V2 46D/6D checkpoint，实际 obs={self.obs_size}, act={self.action_size}"
            )
        if self.verbose:
            log(f"policy ready (obs={self.obs_size}, act={self.action_size}, action_gain={self.action_gain:.2f})")

        self.rng = jax.random.PRNGKey(0)

        if self.control_mode == "observe":
            self.policy_fn = None
        self.hand = HandSerialProtocol(serial_port, serial_baudrate)
        if self.verbose:
            log(f"serial opened: {serial_port} @ {serial_baudrate}")

        self.default_tendon = DEFAULT_TENDON.copy()
        self.action_scale = ACTION_SCALE.copy()
        self.last_action = jp.zeros(6, dtype=jp.float32)
        self.last_action_np = np.zeros(6, dtype=np.float32)
        self.last_raw_action_np = np.zeros(6, dtype=np.float32)
        self.last_motor_targets = self.default_tendon.copy()
        self.last_cmd_deg = ctrl_to_deg(self.default_tendon)
        self.last_readback_deg = np.zeros(6, dtype=np.float32)
        self.last_tactile_pooled_hw = np.zeros(5, dtype=np.float32)
        self.last_tactile_pooled_env = np.zeros(5, dtype=np.float32)
        self.last_tactile_raw = np.zeros((5, 16), dtype=np.uint16)
        self.last_force_safety_scale = 1.0
        self.last_force_balance_obs = 0.0
        self.last_hold_duration_normalized = 0.0
        self.last_state = np.zeros(46, dtype=np.float32)

        self.log_csv_path = log_csv_path
        self._csv_file = None
        self._csv_writer = None
        if self.log_csv_path:
            p = Path(self.log_csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = p.open("w", newline="", encoding="utf-8")
            fields = (
                ["time", "step", "mode", "action_norm", "force_balance_obs", "hold_duration_norm"]
                + [f"cmd_deg_{i}" for i in range(6)]
                + [f"motor_target_{i}" for i in range(6)]
                + [f"raw_action_{i}" for i in range(6)]
                + [f"readback_deg_{i}" for i in range(6)]
                + [f"tactile_hw_{name}" for name in ("thumb", "index", "middle", "ring", "pinky")]
                + [f"tactile_env_{i}" for i in range(5)]
                + ["force_safety_scale", "fb_source", "q_ok", "q_fail"]
                + [f"state_{i}" for i in range(46)]
                + [f"tactile_raw_{f}_{t}" for f in range(5) for t in range(16)]
            )
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fields)
            self._csv_writer.writeheader()
            if self.verbose:
                log(f"csv logging: {p}")

    def _read_hw_pos(self) -> np.ndarray:
        angles_deg = self.hand.query_angles_once()
        if angles_deg is None:
            return self.last_motor_targets.copy()
        self.last_readback_deg = angles_deg.copy()
        safe_k = np.where(np.abs(HW6_K) < 1e-6, 1.0, HW6_K)
        eq_cmd_deg = (angles_deg - HW6_B) / safe_k
        eq_cmd_deg = np.clip(eq_cmd_deg, 0.0, 90.0)
        return deg_to_ctrl(eq_cmd_deg)

    def _build_policy_obs(self) -> dict[str, np.ndarray]:
        _ = self._read_hw_pos()
        hw_pos = self.last_motor_targets.copy()
        self.last_tactile_pooled_hw = self.hand.get_tactile_pooled_hw()
        self.last_tactile_pooled_env = self.hand.get_tactile_pooled_env()
        self.last_tactile_raw = self.hand.get_tactile_raw()
        state, aux = self.obs_projector.build_state(
            motor_targets=hw_pos,
            tactile_pooled_env=self.last_tactile_pooled_env,
            last_action=np.asarray(self.last_action, dtype=np.float32),
        )
        self.last_force_balance_obs = aux["force_balance_obs"]
        self.last_hold_duration_normalized = aux["hold_duration_normalized"]
        self.last_state = state.copy()
        return {"state": state.astype(np.float32)}

    def _rate_limit(self, targets: np.ndarray) -> np.ndarray:
        delta = np.clip(
            targets - self.last_motor_targets,
            -self.max_target_delta,
            self.max_target_delta,
        )
        return self.last_motor_targets + delta

    def _debug_targets_deg(self) -> np.ndarray:
        t = self._step_idx * self.control_dt
        phase = (2.0 * np.pi * t) / self.debug_period_s
        base_deg = ctrl_to_deg(self.default_tendon)
        deg = base_deg.copy()
        for i in range(6):
            wave = np.sin(phase + i * (np.pi / 3.0))
            deg[i] = base_deg[i] + self.debug_amp_deg * wave
        return np.clip(deg, 0.0, 90.0)

    def step_once(self) -> None:
        if self.control_mode == "rl":
            obs = self._build_policy_obs()
            self.rng, act_rng = jax.random.split(self.rng)
            action_jax, _ = self.policy_fn(obs, act_rng)
            action = np.asarray(action_jax, dtype=np.float32).reshape(6)

            self.last_raw_action_np = action.copy()
            action_scaled = action * self.action_gain

            self.last_force_safety_scale = 1.0
            if self.force_safety_threshold > 0:
                max_force = float(np.max(self.last_tactile_pooled_hw))
                if max_force > self.force_safety_threshold:
                    self.last_force_safety_scale = float(
                        np.clip(
                            1.0 - (max_force - self.force_safety_threshold) / self.force_safety_threshold,
                            0.0,
                            1.0,
                        )
                    )
                    action_scaled = action_scaled * self.last_force_safety_scale

            motor_targets = self.default_tendon + action_scaled * self.action_scale
            motor_targets = np.clip(motor_targets, CTRL_RANGES[:, 0], CTRL_RANGES[:, 1])
            motor_targets = self._rate_limit(motor_targets)

            self.last_action = jp.array(action_scaled, dtype=jp.float32)
            self.last_action_np = action_scaled.copy()
            self.last_motor_targets = motor_targets.copy()

            cmd_deg = ctrl_to_deg(motor_targets)
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

        else:
            cmd_deg = self._debug_targets_deg()
            self.last_cmd_deg = cmd_deg.copy()
            self.hand.send_targets_deg(cmd_deg)
            self.last_motor_targets = deg_to_ctrl(cmd_deg)
            self.last_action = jp.zeros(6, dtype=jp.float32)
            self.last_action_np = np.zeros(6, dtype=np.float32)
            self.last_raw_action_np = np.zeros(6, dtype=np.float32)

        self._step_idx += 1

    def _write_csv_row(self, step_count: int) -> None:
        if self._csv_writer is None:
            return
        fb_dbg = self.hand.feedback_debug()
        row: dict[str, Any] = {
            "time": time.time(),
            "step": int(step_count),
            "mode": self.control_mode,
            "action_norm": float(np.linalg.norm(self.last_action_np)),
            "force_balance_obs": float(self.last_force_balance_obs),
            "hold_duration_norm": float(self.last_hold_duration_normalized),
            "fb_source": fb_dbg["source"],
            "q_ok": int(fb_dbg["query_ok"]),
            "q_fail": int(fb_dbg["query_fail"]),
            "force_safety_scale": float(self.last_force_safety_scale),
        }
        for i in range(6):
            row[f"cmd_deg_{i}"] = float(self.last_cmd_deg[i])
            row[f"motor_target_{i}"] = float(self.last_motor_targets[i])
            row[f"raw_action_{i}"] = float(self.last_raw_action_np[i])
            row[f"readback_deg_{i}"] = float(self.last_readback_deg[i])
        for i, name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            row[f"tactile_hw_{name}"] = float(self.last_tactile_pooled_hw[i])
        for i in range(5):
            row[f"tactile_env_{i}"] = float(self.last_tactile_pooled_env[i])
            row[f"state_{6 + i}"] = float(self.last_state[6 + i])
        for i in range(46):
            row[f"state_{i}"] = float(self.last_state[i])
        for f in range(5):
            for t in range(16):
                row[f"tactile_raw_{f}_{t}"] = int(self.last_tactile_raw[f, t])
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def run(self) -> None:
        next_t = time.perf_counter()
        step_count = 0
        try:
            if self.control_mode in ("rl", "debug") and self.preopen_s > 0.0:
                preopen_n = max(1, int(round(self.preopen_s / max(self.control_dt, 1e-3))))
                for _ in range(preopen_n):
                    self.hand.send_open()
                    time.sleep(self.control_dt)
                if self.verbose:
                    log(f"preopen done ({self.preopen_s:.1f}s)")

            if self.control_mode == "observe":
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
                    qangles_txt = (
                        "none"
                        if fb_dbg["last_query_angles_deg"] is None
                        else np.round(fb_dbg["last_query_angles_deg"], 1).tolist()
                    )
                    log(
                        f"step={step_count} "
                        f"cmd_deg={np.round(self.last_cmd_deg, 1).tolist()} "
                        f"action_norm={float(np.linalg.norm(self.last_action_np)):.4f} "
                        f"q_ok={fb_dbg['query_ok']} q_fail={fb_dbg['query_fail']} "
                        f"readback_deg={qangles_txt} "
                        f"tactile_hw={np.round(self.last_tactile_pooled_hw, 1).tolist()} "
                        f"tactile_env={np.round(self.last_tactile_pooled_env, 3).tolist()} "
                        f"force_balance={self.last_force_balance_obs:.3f} "
                        f"hold_norm={self.last_hold_duration_normalized:.3f} "
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


def dry_run_policy(checkpoint_path: str, env_name: str) -> None:
    metadata = _load_qbr_hand_metadata()
    bundle = load_policy_bundle(env_name, checkpoint_path)
    projector = V2ObservationProjector(
        tactile_saturation_n=bundle["env_cfg_dict"]["tactile_config"]["force_saturation_n"],
        hold_success_sec=bundle["env_cfg_dict"]["reward_config"].get("hold_success_sec", 30.0),
    )
    neutral_state, aux = projector.build_state(
        motor_targets=DEFAULT_TENDON.copy(),
        tactile_pooled_env=np.zeros(5, dtype=np.float32),
        last_action=np.zeros(6, dtype=np.float32),
    )
    rng = jax.random.PRNGKey(0)
    action_jax, _ = bundle["policy_fn"]({"state": neutral_state}, rng)
    action = np.asarray(action_jax, dtype=np.float32).reshape(-1)

    log("=== V2 dry run ===")
    log(f"checkpoint_step={bundle['step_dir']}")
    log(f"obs_size={bundle['obs_size']} action_size={bundle['action_size']}")
    log(f"controller_joint_names={list(metadata.controller_joint_names)}")
    log(
        "finger_chains="
        + str({f.name: list(f.joints) for f in (metadata.thumb_chain, *metadata.fingers)})
    )
    log(
        "actuator_joint_map="
        + str({name: joint for name, joint in metadata.actuator_joint_map})
    )
    log(
        "tactile_sites="
        + str({name: len(sites) for name, sites in metadata.tactile_site_map})
    )
    log(f"neutral_hold_norm={aux['hold_duration_normalized']:.3f}")
    log(f"neutral_force_balance={aux['force_balance_obs']:.3f}")
    log(f"policy_action={np.round(action, 5).tolist()}")
    log(
        "占位说明: cube_pos_error/cube_vel/cube_quat/fingertip_to_cube 目前使用中性占位；"
        "真实部署若接入物体位姿估计，可直接替换这 25 维。"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="AeroCubeGraspV2ForceCoacd RL policy -> hardware bridge"
    )
    p.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT_PATH)
    p.add_argument("--serial_port", default=DEFAULT_SERIAL_PORT)
    p.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    p.add_argument("--env_name", default=DEFAULT_ENV_NAME)
    p.add_argument("--control_dt", type=float, default=DEFAULT_CONTROL_DT)
    p.add_argument("--max_target_delta", type=float, default=DEFAULT_MAX_TARGET_DELTA)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--heartbeat_steps", type=int, default=20)
    p.add_argument("--mode", choices=["rl", "debug", "observe"], default="rl")
    p.add_argument("--debug_amp_deg", type=float, default=8.0)
    p.add_argument("--debug_period_s", type=float, default=2.5)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--log_csv", default=DEFAULT_LOG_CSV)
    p.add_argument("--preopen_s", type=float, default=DEFAULT_PREOPEN_S)
    p.add_argument("--action_gain", type=float, default=DEFAULT_ACTION_GAIN)
    p.add_argument(
        "--force_safety_threshold",
        type=float,
        default=DEFAULT_FORCE_SAFETY_THRESHOLD,
        help="硬件原始 pooled tactile 超阈值时衰减动作 (0=禁用)",
    )
    p.add_argument(
        "--dry_run_policy",
        action="store_true",
        help="不连接串口，只恢复 checkpoint、打印 URDF 元数据并跑一次中性观测推理。",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log("=== AeroCubeGraspV2ForceCoacd Hardware Bridge ===")
    log(f"checkpoint={args.checkpoint_path}")
    log(f"env={args.env_name}")
    if args.dry_run_policy:
        dry_run_policy(args.checkpoint_path, args.env_name)
        return

    bridge = AeroV2Bridge(
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

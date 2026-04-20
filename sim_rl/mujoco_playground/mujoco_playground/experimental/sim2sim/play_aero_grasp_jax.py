# Copyright 2026
#
# AeroCubeGrasp sim2sim 部署脚本：
# 加载 PPO checkpoint，并在原生 MuJoCo viewer 中进行实时控制。

import argparse
import functools

from etils import epath
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer as viewer
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import base as aero_hand_base
from mujoco_playground.config import manipulation_params


def _load_policy(env_name: str, checkpoint_path: str):
  # 按环境配置重建与训练一致的 PPO 网络结构，确保 checkpoint 参数可正确匹配。
  env = registry.load(env_name)
  ppo_params = manipulation_params.brax_ppo_config(env_name)
  network_factory_config = ppo_params.get("network_factory", {})
  del ppo_params["network_factory"]

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **network_factory_config
  )

  if "num_timesteps" in ppo_params:
    del ppo_params["num_timesteps"]

  # 通过 ppo_train 的恢复路径加载 Orbax 目录格式 checkpoint。
  make_inference_fn, params, _ = ppo_train(
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      network_factory=network_factory,
      num_timesteps=0,
      seed=1,
      restore_checkpoint_path=epath.Path(checkpoint_path).resolve(),
      **ppo_params,
  )

  # 保留完整 tuple；Brax PPO 策略内部期望使用 params[0] 和 params[1]。
  if isinstance(params, dict):
    # 兼容少数非标准 dict 结构 checkpoint（尽力而为）。
    normalizer = params.get("normalizer") or params.get("normalizer_params")
    policy = params.get("policy") or params.get("policy_params") or params.get("params")
    if normalizer is not None and policy is not None:
      params = (normalizer, policy)

  # 部署时默认使用确定性推理，便于结果复现。
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  return jit_inference_fn


class AeroGraspJaxController:
  """AeroCubeGrasp 在原生 MuJoCo 中的实时控制器。"""

  def __init__(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      checkpoint_path: str,
      env_name: str = "AeroCubeGrasp",
      history_len: int = 1,
      release_after_sec: float = 1.2,
      support_pos: tuple[float, float, float] = (-0.066, 0.0, 0.067),
      support_hidden_pos: tuple[float, float, float] = (0.0, 0.0, -10.0),
  ):
    # 策略函数：输入观测字典，输出动作。
    self._inference_fn = _load_policy(env_name, checkpoint_path)
    self._rng = jax.random.PRNGKey(0)

    self._model = model
    self._data = data
    # 必须与训练环境中的 action_scale 一致，否则行为会偏移。
    self._action_scale = jp.array(
        [0.02, 0.02, 0.02, 0.02, 0.7, 0.003, 0.012], dtype=jp.float32
    )

    # 控制频率 20 Hz，仿真积分频率 100 Hz。
    self._ctrl_dt = 0.05
    self._sim_dt = 0.01
    self._n_substeps = max(1, int(round(self._ctrl_dt / self._sim_dt)))
    self._counter = 0

    # state = 6 路腱长传感 + 1 路拇指外展传感 + 7 路 last_action = 14 维。
    self._obs_dim = 14
    self._history_len = history_len
    self._obs_history = jp.zeros((self._history_len * self._obs_dim,), dtype=jp.float32)
    self._last_action = jp.zeros((consts.NU,), dtype=jp.float32)

    # 使用 home keyframe 的 ctrl 作为腱驱动基线，与训练 reset 保持一致。
    home_kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_kf_id < 0:
      raise ValueError("home keyframe is required for AeroCubeGrasp deployment")
    self._default_tendon = jp.array(model.key_ctrl[home_kf_id], dtype=jp.float32)

    # 可选支撑体逻辑：延时后隐藏，行为与训练环境一致。
    self._support_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "cube_support"
    )
    self._support_mocap_id = -1
    if self._support_body_id >= 0:
      self._support_mocap_id = int(model.body_mocapid[self._support_body_id])
    self._support_pos = np.array(support_pos, dtype=np.float32)
    self._support_hidden_pos = np.array(support_hidden_pos, dtype=np.float32)
    self._release_after_steps = max(1, int(round(release_after_sec / self._ctrl_dt)))
    self._support_timer = 0
    self._support_released = False

    # 同步 MuJoCo 步长并用 home 姿态初始化。
    model.opt.timestep = self._sim_dt
    mujoco.mj_resetDataKeyframe(model, data, home_kf_id)

    if self._support_mocap_id >= 0 and model.nmocap > 0:
      data.mocap_pos[self._support_mocap_id] = self._support_pos

  def _read_obs_state(self) -> jp.ndarray:
    # 观测构造必须与 grasp_cube.py 中策略使用的 state 完全一致。
    tendon_lengths = []
    for name in consts.SENSOR_TENDON_NAMES:
      tendon_lengths.append(np.array(self._data.sensor(name).data).reshape(-1)[0])

    joint_values = []
    for name in consts.SENSOR_JOINT_NAMES:
      joint_values.append(np.array(self._data.sensor(name).data).reshape(-1)[0])

    state = jp.array(tendon_lengths + joint_values, dtype=jp.float32)
    state = jp.concatenate([state, self._last_action])

    # 历史堆叠：最新帧放前面，旧帧后移。
    self._obs_history = jp.roll(self._obs_history, state.size)
    self._obs_history = self._obs_history.at[: state.size].set(state)
    return self._obs_history

  def _update_support(self) -> None:
    if self._support_mocap_id < 0 or self._model.nmocap == 0:
      return
    self._support_timer += 1
    if (not self._support_released) and self._support_timer >= self._release_after_steps:
      self._support_released = True
      self._data.mocap_pos[self._support_mocap_id] = self._support_hidden_pos

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    del model, data
    self._counter += 1
    # 仅在控制时刻运行策略；MuJoCo 仍按每个 sim_dt 连续步进。
    if self._counter % self._n_substeps != 0:
      return

    self._update_support()

    obs_state = self._read_obs_state()
    # 部署时 actor 只使用 "state"；critic 专用输入在部署中不可用。
    policy_obs = {"state": np.asarray(obs_state, dtype=np.float32)}

    self._rng, act_rng = jax.random.split(self._rng)
    action_jax, _ = self._inference_fn(policy_obs, act_rng)
    action = jp.array(action_jax, dtype=jp.float32)

    # 与训练动作映射一致：target = default + scaled_action。
    motor_targets = self._default_tendon + action * self._action_scale

    ctrl_min = jp.array(self._model.actuator_ctrlrange[:, 0], dtype=jp.float32)
    ctrl_max = jp.array(self._model.actuator_ctrlrange[:, 1], dtype=jp.float32)
    motor_targets = jp.clip(motor_targets, ctrl_min, ctrl_max)

    # 将腱目标值写入执行器控制量。
    self._data.ctrl[:] = np.asarray(motor_targets, dtype=np.float32)
    self._last_action = action


def _make_model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
  xml_path = consts.GRASP_XML
  model = mujoco.MjModel.from_xml_string(
      epath.Path(xml_path).read_text(), assets=aero_hand_base.get_assets()
  )
  data = mujoco.MjData(model)
  return model, data


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--checkpoint_path",
      required=True,
        help="checkpoint 目录路径（如 logs/*/checkpoints/000xxxxxx）",
  )
  parser.add_argument(
      "--env_name",
      default="AeroCubeGrasp",
        help="在 mujoco_playground registry 中注册的环境名",
  )
  parser.add_argument(
      "--history_len",
      default=1,
      type=int,
        help="策略训练使用的观测历史长度",
  )
  parser.add_argument(
      "--release_after_sec",
      default=1.2,
      type=float,
        help="仿真中支撑体 mocap 延时隐藏的秒数",
  )
  args = parser.parse_args()

  model, data = _make_model_and_data()
  controller = AeroGraspJaxController(
      model=model,
      data=data,
      checkpoint_path=args.checkpoint_path,
      env_name=args.env_name,
      history_len=args.history_len,
      release_after_sec=args.release_after_sec,
  )

  mujoco.set_mjcb_control(controller.get_control)
  try:
    viewer.launch(model, data)
  finally:
    mujoco.set_mjcb_control(None)


if __name__ == "__main__":
  main()

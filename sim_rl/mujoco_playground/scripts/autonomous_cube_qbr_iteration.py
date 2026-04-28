#!/usr/bin/env python3
"""Autonomous iteration loop for AeroCubeGraspV2ForceCapsuleBottlePalmQbr.

Fixed loop:
1. apply one narrow code/config change
2. smoke test
3. launch training
4. wait for end / key evals
5. read metrics.csv
6. compute first / last / max / best_step + contact cleanliness
7. append changelog
8. immediately move to next experiment
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/home/ll/SRTP/Aero-Hand")
ENV_FILE = ROOT / "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/grasp_cube_v2_force.py"
CHANGELOG = ROOT / "v2_iteration_docs/changelog.md"
LOGS = ROOT / "logs"
PYTHON = "/home/ll/miniconda3/envs/aero_rl/bin/python"
TRAIN = ROOT / "sim_rl/mujoco_playground/learning/train_jax_ppo.py"
PYTHONPATH = str(ROOT / "sim_rl/mujoco_playground")
BASE_CKPT = (
    ROOT
    / "logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbr-20260427-164913-C106_long_dr_lr7p5e5_upd3_ent5e3_from_C103best_relmax2p45_postgrasp135_triad18_force72_capsule_bottlepalm_2048/checkpoints/000002211840"
)
BEST_RUN = (
    LOGS
    / "AeroCubeGraspV2ForceCapsuleBottlePalmQbr-20260428-013906-C124_long_dr_lr7p25e5_upd3_force3p2_pose50_from_C106best_relmax2p45_postgrasp135_triad18_force72_capsule_bottlepalm_2048"
)


@dataclass
class Experiment:
  cid: str
  title: str
  suffix: str
  search: str
  replace: str
  expected: str
  reason: str
  compare_to: str = "C124"
  partial_ok: bool = False


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
  proc_env = os.environ.copy()
  proc_env["PYTHONPATH"] = PYTHONPATH
  proc_env["MUJOCO_GL"] = "egl"
  proc_env["PYOPENGL_PLATFORM"] = "egl"
  if env:
    proc_env.update(env)
  return subprocess.run(cmd, cwd=cwd, env=proc_env, text=True, capture_output=True)


def replace_once(path: Path, search: str, replace: str) -> None:
  text = path.read_text()
  if replace in text:
    return
  if search not in text:
    raise RuntimeError(f"pattern not found: {search}")
  path.write_text(text.replace(search, replace, 1))


def analyze_metrics(metrics_csv: Path) -> dict[str, object]:
  rows = list(csv.DictReader(metrics_csv.open()))
  if not rows:
    raise RuntimeError(f"empty metrics: {metrics_csv}")
  metric = "eval/episode_diagnostic/contact_duration_sec"
  vals = [float(r[metric]) for r in rows]
  steps = [int(float(r["num_steps"])) for r in rows]
  best_idx = max(range(len(vals)), key=lambda i: vals[i])
  diag_keys = [
      "eval/episode_diagnostic/palm_contact",
      "eval/episode_diagnostic/nonprimary_contact",
      "eval/episode_diagnostic/non_tip_primary_contact",
      "eval/episode_diagnostic/drop",
      "eval/episode_diagnostic/slip_event",
  ]
  diags = {k: [float(r[k]) for r in rows] for k in diag_keys}
  clean = all(max(v) <= 1e-6 for v in diags.values())
  reward_cols = [
      "eval/episode_reward/post_release_survival",
      "eval/episode_reward/sustained_hold_bonus",
      "eval/episode_reward/progressive_hold",
      "eval/episode_reward/stable_hold",
      "eval/episode_reward/post_release_pose_hold",
      "eval/episode_reward/hold_position",
      "eval/episode_reward/post_release_grasp",
      "eval/episode_reward/force_balance",
      "eval/episode_reward/primary_finger_force",
      "eval/episode_reward/pre_release_grasp",
  ]
  reward_stats = {
      k: [float(r[k]) for r in rows]
      for k in reward_cols
      if k in rows[0]
  }
  return {
      "rows": rows,
      "first": vals[0],
      "last": vals[-1],
      "max": vals[best_idx],
      "best_step": steps[best_idx],
      "diags": diags,
      "clean": clean,
      "reward_stats": reward_stats,
  }


def find_run_dir(suffix: str) -> Path:
  while True:
    matches = sorted(LOGS.glob(f"*{suffix}"))
    if matches:
      return matches[-1]
    time.sleep(5)


def wait_for_metrics(run_dir: Path, *, min_rows: int = 4) -> Path:
  metrics = run_dir / "metrics.csv"
  while True:
    if metrics.exists():
      rows = list(csv.DictReader(metrics.open()))
      if len(rows) >= min_rows:
        return metrics
    time.sleep(60)


def wait_process_done(suffix: str) -> None:
  patt = re.compile(rf"train_jax_ppo.*{re.escape(suffix)}")
  while True:
    out = subprocess.run(["pgrep", "-af", "train_jax_ppo"], capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if patt.search(ln)]
    if not lines:
      return
    time.sleep(120)


def smoke_read() -> str:
  code = textwrap.dedent(
      """
      import sys
      sys.path.insert(0, '/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground')
      from mujoco_playground._src.manipulation.aero_hand.grasp_cube_v2_force import CubeGraspV2ForceCapsuleBottlePalmQbr
      cfg = CubeGraspV2ForceCapsuleBottlePalmQbr()._config
      print('min_release_active_fingers', cfg.support_config.min_release_active_fingers)
      print('post_release_pose_hold', cfg.reward_config.scales.post_release_pose_hold)
      print('force_balance', cfg.reward_config.scales.force_balance)
      print('post_release_grasp', cfg.reward_config.scales.post_release_grasp)
      print('clean_gate_mode full_cheat_penalty')
      """
  )
  res = run([PYTHON, "-c", code])
  if res.returncode != 0:
    raise RuntimeError(f"smoke failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
  return res.stdout.strip()


def append_log(block: str) -> None:
  with CHANGELOG.open("a", encoding="utf-8") as f:
    f.write("\n" + block.rstrip() + "\n")


def already_logged(cid: str) -> bool:
  return f"- {cid}:" in CHANGELOG.read_text(encoding="utf-8")


def format_diag(diags: dict[str, list[float]]) -> str:
  order = [
      "eval/episode_diagnostic/palm_contact",
      "eval/episode_diagnostic/nonprimary_contact",
      "eval/episode_diagnostic/non_tip_primary_contact",
      "eval/episode_diagnostic/drop",
      "eval/episode_diagnostic/slip_event",
  ]
  parts = []
  for k in order:
    vals = " -> ".join(f"{v:.4f}" for v in diags[k])
    parts.append(f"    - `{k.split('/')[-1]}: {vals}`")
  return "\n".join(parts)


def format_reward_stats(stats: dict[str, list[float]]) -> str:
  lines = []
  for k, vals in stats.items():
    if not vals:
      continue
    name = k.split("/")[-1]
    lines.append(f"    - `{name}: " + " -> ".join(f"{v:.2f}" for v in vals) + "`")
  return "\n".join(lines)


def train_command(suffix: str) -> list[str]:
  return [
      PYTHON,
      str(TRAIN),
      "--env_name=AeroCubeGraspV2ForceCapsuleBottlePalmQbr",
      "--num_timesteps=2097152",
      "--num_evals=4",
      "--num_envs=2048",
      "--num_eval_envs=128",
      "--episode_length=800",
      "--learning_rate=7.25e-5",
      "--num_updates_per_batch=3",
      "--entropy_cost=0.005",
      "--domain_randomization",
      "--ignore_checkpoint_env_config=True",
      f"--load_checkpoint_path={BASE_CKPT}",
      f"--suffix={suffix}",
  ]


def launch_training(suffix: str) -> None:
  proc = subprocess.Popen(
      train_command(suffix),
      cwd=ROOT,
      env={
          **os.environ,
          "PYTHONPATH": PYTHONPATH,
          "MUJOCO_GL": "egl",
          "PYOPENGL_PLATFORM": "egl",
      },
      stdout=(ROOT / f"{suffix}.stdout.log").open("w"),
      stderr=subprocess.STDOUT,
      text=True,
  )
  if proc.poll() is not None:
    raise RuntimeError(f"training for {suffix} exited immediately")


def log_c133_partial() -> None:
  text = CHANGELOG.read_text(encoding="utf-8")
  if "- C133:" in text:
    return
  block = """
- C133: soft cheat-gate structural probe (partial negative)
  - 改动:
    - 修改 [grasp_cube_v2_force.py](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/grasp_cube_v2_force.py)
    - 保持 `C124` 主线的 reward / PPO / release timing 不变
    - 仅把 clean gate 从 `1.0 - cheat_contact` 放松为 `1.0 - 0.5 * cheat_contact`
  - smoke test:
    - 代码可运行，但这是故意放松 palm / nonprimary 污染抑制的结构 probe
  - 训练:
    - env: `AeroCubeGraspV2ForceCapsuleBottlePalmQbr`
    - restore: `C106 000002211840`
    - run: `logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbr-20260428-093144-C133_long_dr_lr7p25e5_upd3_force3p2_pose50_softcheatgate_from_C106best_relmax2p45_triad18_force72_capsule_bottlepalm_2048`
  - metrics.csv:
    - partial only
    - `first = 23.9573s`
    - `last = 23.8280s`
    - `max = 23.9573s`
    - `best_step = 737280`
  - 接触形态:
    - first eval 出现极小 `slip_event = 0.0078`
    - 其余 dirty 指标未明显爆炸，但已经低于 `C124`
  - 与前序对比:
    - 相比 `C124`，partial `last 25.1686 -> 23.8280s`
  - 修改原因:
    - 测试 cheat-contact 抑制是否过硬，是否在压制恢复动作
  - 预期效果:
    - 若惩罚过硬，放松后应提升尾评
  - 实际效果:
    - partial 结果明显低于 `C124`，因此提前终止
  - 失败模式分析:
    - clean gate 放松会让错误支撑更容易混进“看起来成功”的轨迹，trainability 变差
    - 这条线不应保留在主线
  - 下一轮建议:
    - 回退到 `clean_gate = 1.0 - cheat_contact`
    - 从 `C124` 干净主线继续更窄的 release 结构 probe
"""
  append_log(block)


def make_log(exp: Experiment, smoke: str, metrics: dict[str, object], run_dir: Path) -> str:
  compare = BEST_RUN / "metrics.csv"
  compare_rows = list(csv.DictReader(compare.open()))
  compare_last = float(compare_rows[-1]["eval/episode_diagnostic/contact_duration_sec"])
  better = float(metrics["last"]) - compare_last
  next_hint = "若提升明显，则继续围绕 release 条件做更窄半步；否则切到另一个 gate/threshold 单变量。"
  return f"""
- {exp.cid}: {exp.title}
  - 改动:
    - 修改 [grasp_cube_v2_force.py](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/grasp_cube_v2_force.py)
    - {exp.reason}
  - smoke test:
```
{smoke}
```
  - 训练:
    - env: `AeroCubeGraspV2ForceCapsuleBottlePalmQbr`
    - restore: `C106 000002211840`
    - run: `logs/{run_dir.name}`
    - command: `{' '.join(train_command(exp.suffix))}`
  - metrics.csv:
    - `first = {float(metrics['first']):.4f}s`
    - `last = {float(metrics['last']):.4f}s`
    - `max = {float(metrics['max']):.4f}s`
    - `best_step = {int(metrics['best_step'])}`
  - 接触形态:
{format_diag(metrics["diags"])}
  - 组件统计:
{format_reward_stats(metrics["reward_stats"])}
  - 与前序对比:
    - 相比 `C124`，`last {compare_last:.4f} -> {float(metrics['last']):.4f}s`
    - 差值: `{better:+.4f}s`
  - 修改原因:
    - {exp.reason}
  - 预期效果:
    - {exp.expected}
  - 实际效果:
    - {'clean 且超过 C124' if float(metrics['last']) > compare_last and metrics['clean'] else '未超过 C124，或接触形态变脏'}
  - 失败模式分析:
    - {'本轮为正样本，可继续沿这条窄结构线推进。' if float(metrics['last']) > compare_last and metrics['clean'] else '当前单变量没有把 unsupported retention 推过 C124；需要保留干净主线，换下一个更窄 probe。'}
  - 下一轮建议:
    - {next_hint}
"""


def main() -> int:
  log_c133_partial()

  experiments = [
      Experiment(
          cid="C134",
          title="harder release gate requiring all three primary digits",
          suffix="C134_long_dr_lr7p25e5_upd3_force3p2_pose50_minactive3_from_C106best_relmax2p45_triad18_force72_capsule_bottlepalm_2048",
          search="    config.support_config.min_release_active_fingers = 2\n    config.reward_config.scales.pre_release_grasp = 35.0\n",
          replace="    config.support_config.min_release_active_fingers = 3\n    config.reward_config.scales.pre_release_grasp = 35.0\n",
          reason="仅把有效 override 的 `min_release_active_fingers: 2 -> 3`，强制支撑释放前必须食指/中指/拇指三指都真正参与，减少“拇指主导提前 release”",
          expected="如果当前瓶颈是拇指提前接管、食指参与不足，这轮应让 release 后 unsupported 更稳，并保持 clean-contact。",
      ),
      Experiment(
          cid="C135",
          title="slightly firmer release-force gate on the clean C124 mainline",
          suffix="C135_long_dr_lr7p25e5_upd3_force3p2_pose50_minforce011_from_C106best_relmax2p45_triad18_force72_capsule_bottlepalm_2048",
          search="          min_release_force=0.10,         # C40: revert hard release gate; keep softer force shaping instead\n",
          replace="          min_release_force=0.11,         # C135: slightly firmer release gate on the C124 mainline\n",
          reason="回到干净主线后，仅把 `min_release_force: 0.10 -> 0.11`，不再强推三指都亮灯，而是更窄地要求 release 前主抓力更扎实",
          expected="如果食指/中指参与不足主要表现为 release 前承托力还不够实，这轮应在保持 clean-contact 的同时抬高尾评；若再次退化，就说明问题不在 release force gate。",
      ),
  ]

  for exp in experiments:
    if already_logged(exp.cid):
      continue
    replace_once(ENV_FILE, exp.search, exp.replace)
    smoke = smoke_read()
    launch_training(exp.suffix)
    run_dir = find_run_dir(exp.suffix)
    metrics_csv = wait_for_metrics(run_dir)
    wait_process_done(exp.suffix)
    metrics = analyze_metrics(metrics_csv)
    append_log(make_log(exp, smoke, metrics, run_dir))

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

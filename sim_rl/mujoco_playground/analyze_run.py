#!/usr/bin/env python3
"""Analyze or wait on Aero hand PPO training metrics.

Usage:
  python analyze_run.py <logdir>
  python analyze_run.py <logdir> --wait-rows 5 --poll-sec 60
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def _read_rows(logdir: Path) -> list[dict[str, str]]:
    csv_path = logdir / "metrics.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def _i(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except ValueError:
        return default


def _rate(row: dict[str, str], key: str) -> float:
    ep_len = max(_f(row, "eval/avg_episode_length"), 1.0)
    return _f(row, key) / ep_len


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def print_table(rows: list[dict[str, str]], tail: int | None = None) -> None:
    if not rows:
        print("No metrics rows yet.")
        return
    shown = rows[-tail:] if tail else rows
    print(
        f"{'Step':>10} {'EpLen':>6} {'Reward':>8} "
        f"{'Palm/s':>7} {'3f/s':>7} {'2+f/s':>7} {'Prim/s':>7} {'NTip/s':>7} "
        f"{'Lift/s':>7} {'Hold/s':>7} {'LInit':>6} {'Grip':>7} {'PFF':>7} "
        f"{'HoldR':>7} {'Stable':>7} {'FBal':>7} {'Drop%':>6}"
    )
    print("-" * 140)
    for row in shown:
        print(
            f"{_i(row, 'num_steps'):>10} "
            f"{_f(row, 'eval/avg_episode_length'):>6.1f} "
            f"{_f(row, 'eval/episode_reward'):>8.1f} "
            f"{_fmt_pct(_rate(row, 'eval/episode_diagnostic/palm_contact')):>7} "
            f"{_fmt_pct(_rate(row, 'eval/episode_diagnostic/three_finger_contact')):>7} "
            f"{_fmt_pct(_rate(row, 'eval/episode_diagnostic/two_plus_primary_contact')):>7} "
            f"{_rate(row, 'eval/episode_diagnostic/primary_active_count'):>7.2f} "
            f"{_fmt_pct(_rate(row, 'eval/episode_diagnostic/non_tip_primary_contact')):>7} "
            f"{_fmt_pct(_rate(row, 'eval/episode_diagnostic/lift_success')):>7} "
            f"{_fmt_pct(_rate(row, 'eval/episode_diagnostic/hold_success')):>7} "
            f"{_fmt_pct(_rate(row, 'eval/episode_diagnostic/lifted_reset')):>6} "
            f"{_f(row, 'eval/episode_reward/grip_force'):>7.2f} "
            f"{_f(row, 'eval/episode_reward/primary_finger_force'):>7.3f} "
            f"{_f(row, 'eval/episode_reward/hold_position'):>7.3f} "
            f"{_f(row, 'eval/episode_reward/stable_hold'):>7.3f} "
            f"{_f(row, 'eval/episode_reward/force_balance'):>7.1f} "
            f"{100.0 * _f(row, 'eval/episode_termination/drop'):>5.0f}%"
        )


def print_summary(rows: list[dict[str, str]], total_steps: int) -> None:
    if not rows:
        return
    first, last = rows[0], rows[-1]
    print("\n--- Summary ---")
    print(
        f"Reward: {_f(first, 'eval/episode_reward'):.1f} -> "
        f"{_f(last, 'eval/episode_reward'):.1f} "
        f"(delta={_f(last, 'eval/episode_reward') - _f(first, 'eval/episode_reward'):+.1f})"
    )
    print(
        f"EpLen:  {_f(first, 'eval/avg_episode_length'):.1f} -> "
        f"{_f(last, 'eval/avg_episode_length'):.1f}"
    )
    peak = max(rows, key=lambda r: _f(r, "eval/episode_reward"))
    print(
        f"Peak reward: {_f(peak, 'eval/episode_reward'):.1f} "
        f"at step {_i(peak, 'num_steps')}"
    )

    step = _i(last, "num_steps")
    sps = _f(last, "eval/sps")
    if sps > 0 and step < total_steps:
        remaining_s = (total_steps - step) / sps
        print(f"ETA at current eval/sps: {remaining_s / 60.0:.1f} min")


def wait_for_rows(
    logdir: Path,
    wait_rows: int,
    poll_sec: float,
    max_minutes: float | None,
    total_steps: int,
) -> list[dict[str, str]]:
    start = time.monotonic()
    last_count = -1
    while True:
        rows = _read_rows(logdir)
        if len(rows) != last_count:
            last_count = len(rows)
            print(f"\n[{time.strftime('%H:%M:%S')}] rows={len(rows)}")
            print_table(rows, tail=min(5, len(rows)))
            print_summary(rows, total_steps)
            sys.stdout.flush()
        if len(rows) >= wait_rows:
            return rows
        if max_minutes is not None and (time.monotonic() - start) > max_minutes * 60.0:
            print(f"Timed out after {max_minutes:.1f} min waiting for rows={wait_rows}.")
            return rows
        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=Path)
    parser.add_argument("--tail", type=int, default=0, help="Only print last N rows.")
    parser.add_argument("--wait-rows", type=int, default=0, help="Wait until metrics has at least N rows.")
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=40_000_000)
    args = parser.parse_args()

    if not args.logdir.exists():
        print(f"Logdir does not exist: {args.logdir}", file=sys.stderr)
        return 2

    if args.wait_rows:
        rows = wait_for_rows(
            args.logdir, args.wait_rows, args.poll_sec,
            args.max_minutes, args.total_steps,
        )
    else:
        rows = _read_rows(args.logdir)
        print_table(rows, tail=args.tail or None)
        print_summary(rows, args.total_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

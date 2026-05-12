# Cube Iteration Value Summary (2026-04-28)

Scope: `AeroCubeGraspV2ForceCapsuleBottlePalmQbr` around the smoothed thumb collision migration and recovery path from `C124` to `C157`.

## Bottom Line

The highest-value change was not another reward tweak. It was restoring the cube/support `x` alignment toward the older cube line after the thumb collision became smoother.

Current validated mainline:

- env: `AeroCubeGraspV2ForceCapsuleBottlePalmQbr`
- config in [grasp_cube_v2_force.py](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/grasp_cube_v2_force.py:113)
- `support_pos = [0.021, -0.065, 0.1308]`
- `cube_pos = [0.021, -0.065, 0.1463]`
- `pre_grasp_noise_scale = 0.10`
- validated run: `C157`
- `first = 37.3521s`
- `last = 36.6489s`
- `max = 37.3521s`
- `palm/nonprimary/drop = 0`

## Ranked High-Value Changes

### 1. `x` realignment to `0.021` was the decisive recovery

- `C154 last = 17.0668s`
- `C156 last = 37.0044s`
- `C157 last = 36.6489s`
- gain from `C154 -> C156`: `+19.9376s`
- gain from old-geometry baseline `C124 -> C157`: `+11.4803s`

Why it mattered:

- The smoothed thumb collision did not just shift a scalar reward optimum.
- It changed the contact proxy enough that the old cube seating line was no longer the best match.
- Moving the cube/support pair back toward the older `x` line recovered a much better thumb-under / index-led closure.

Evidence:

- `C153/C154` already showed `x - 2mm` was strongly positive.
- `C156` showed `x - 4mm` was the step change, not a small marginal gain.
- `C157` confirmed the gain was stable under long validation, not just a short-probe artifact.

### 2. Smoothed thumb collision broke old policy transfer and forced a geometry re-search

- old baseline `C124 last = 25.1686s`
- immediate new-geometry validation `C136 last = 10.6516s`
- transfer loss: `-14.5171s`

Why it mattered:

- This was the key diagnostic event.
- It proved the new collision fit was geometrically better relative to STL, but no longer compatible with the old learned contact proxy.
- That prevented wasting time on reward-only explanations.

### 3. Lowering seated height by `4mm` was a useful but limited precursor

- `C136 last = 10.6516s`
- `C138 last = 12.2055s` with `z - 2mm`
- `C139 last = 12.3805s` with `z - 4mm`
- `C140 last = 8.6672s` with `z - 6mm`

Why it mattered:

- It established that the new thumb geometry wanted a lower seat.
- It also established the usable range: `-4mm` helped, `-6mm` broke the grasp and dirtied contact.
- That gave a clean geometric branch to continue from before the larger `x` discovery.

### 4. Restoring from `C124` best was better than restoring from the older `C106` branch

- `C139 last = 12.3805s`
- `C141 last = 12.7047s`
- gain: `+0.3242s`

Why it mattered:

- The newer checkpoint carried a better post-release scaffold into the new geometry.
- This was not a full fix, but it moved the search onto a stronger branch and made later improvements easier to observe.

### 5. `learning_rate = 5e-5` and `pre_grasp_noise_scale = 0.10` were the best trainability settings on the smoothed geometry branch

- `C141 last = 12.7047s`
- `C142 last = 12.7824s`
- `C144 last = 12.8461s`
- `C146 last = 13.1735s`

Why it mattered:

- These gains were small, but they were clean and repeatable.
- They turned the post-smoothed-geometry branch from a flat `~10-12s` zone into a stable `~13s` branch.
- That branch became the launch point for the later `x` realignment sweep.

### 6. Reward subtraction at `C124` remained the strongest old-geometry reward change

- `C124` only changed `post_release_pose_hold: 60 -> 50`
- `C124 last = 25.1686s`
- all dirty-contact diagnostics remained zero

Why it mattered:

- It was the last clean old-geometry baseline.
- More importantly, it showed that reducing over-strong pose preservation was better than inflating long-hold bonuses.
- Later migration work was measured against this clean baseline, not against noisier reward experiments.

## High-Value Negative Results

These runs did not improve the metric, but they saved time by closing bad branches early.

### `C140`: `z - 6mm` over-lowered the seat

- `last = 8.6672s`
- `palm_contact = 5.1484`
- `nonprimary_contact = 0.0234`
- `drop = 0.2734`

Value:

- proved the lower-seat branch had already crossed the useful range

### `C145`: tighter reset noise to `0.05` was worse than `0.10`

- `C144 last = 12.8461s`
- `C145 last = 12.7957s`

Value:

- prevented over-tightening the curriculum and losing adaptation signal

### `C147` and `C155`: continuation from the latest best checkpoint was unstable

- `C146 last = 13.1735s`
- `C147 last = 12.3457s`
- `C154 last = 17.0668s`
- `C155 last = 15.7379s`

Value:

- showed that continuing from the newest local best was less reliable than restoring from the stable `C124` branch and sweeping geometry

## Selected Runs

| Run | Main change | first (s) | last (s) | max (s) | Note |
| --- | --- | ---: | ---: | ---: | --- |
| `C124` | reward subtraction on `post_release_pose_hold` | 24.1273 | 25.1686 | 25.1686 | old-geometry clean baseline |
| `C136` | smoothed thumb collision validation | 10.7887 | 10.6516 | 10.7887 | transfer collapse |
| `C139` | `z - 4mm` | 12.6336 | 12.3805 | 12.6336 | best height-only point |
| `C141` | restore from `C124` best | 12.9469 | 12.7047 | 12.9469 | better restore source |
| `C142` | `lr = 5e-5` | 12.9824 | 12.7824 | 12.9824 | better trainability |
| `C144` | `pre_grasp_noise = 0.10` | 12.7887 | 12.8461 | 12.8461 | best short probe on this branch |
| `C146` | long validate `C144` branch | 12.8391 | 13.1735 | 13.1735 | stable precursor branch |
| `C153` | `x - 2mm` | 16.7488 | 15.9742 | 16.7488 | first strong geometric recovery |
| `C154` | long validate `x - 2mm` | 16.6715 | 17.0668 | 17.0668 | stable but still below old baseline |
| `C156` | `x - 4mm` | 37.3778 | 37.0044 | 37.3778 | decisive recovery |
| `C157` | long validate `x - 4mm` | 37.3521 | 36.6489 | 37.3521 | current mainline |

## Practical Takeaways

1. After a collision-fit change, first suspect contact geometry alignment, not reward.
2. Use the clean old mainline checkpoint as the restore anchor when migration continuity matters.
3. Height scans can recover part of the lost scaffold, but they may only be a precursor.
4. When a geometric branch starts to work, do not assume continuation from the newest best is safer.
5. For this migration, `x` alignment dominated every other single variable.

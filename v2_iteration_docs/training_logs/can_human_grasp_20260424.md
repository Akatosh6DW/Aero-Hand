# CAN Human-Grasp Iteration Notes (2026-04-24)

Object pose and hand base stayed fixed to the user-provided setup:

- object pos: `[0.008041, -0.040830, 0.132128]`
- object quat_wxyz: `[0.707715, -0.007412, 0.706458, 0.000577]`
- hand base: unchanged

Goal used for these iterations:

- human-like can grasp
- finger joints press the can into the palm
- thumb supports from below/front instead of pure side pinch
- support may guide, but reward must not come mainly from platform support
- DR remains off until the unsupported hold is genuinely stable

## CAN52 `jointpalm_thumbunder_isolatedcfg`

Changes:

- isolated checkpoint loading so old env config no longer overwrites new reward settings
- added `joint_palm_clamp`
- rewrote thumb reward toward under-support
- reduced old tripod bias

Result:

- reward rose from `2320.680` to `2371.188`
- grasp shape improved, but unsupported retention was still poor
- `drop` stayed at `1.0`

## CAN53 `ring_palm_ulnar`

Changes:

- default ctrl moved to a contact-feasible posture that gives palm + middle + ring + thumb contact
- ring/ulnar-side reward weights increased
- support-phase reward reduced further
- post-release retention terms increased

Result:

- reward: `13375.348 -> 25399.330`
- `contact_duration_sec`: `4.43 -> 8.66`
- `joint_palm_contact`: `87.09 -> 173.01`
- `ring_wrap_contact`: `145.91 -> 173.76`
- `palm_contact`: `174.16 -> 199.24`
- still not at target; `drop` remained `1.0`

Interpretation:

- policy now learns the intended human-like wrap much better
- main remaining gap is turning strong supported wrap into post-release retention

## CAN54 `release_gate_tighten`

Changes:

- release support later and more slowly:
  - `release_after_sec = 9.0`
  - `release_ramp_sec = 1.4`
  - `force_release_after_sec = 12.0`
- stronger release requirement:
  - `min_release_force = 0.11`
- support rewards reduced again:
  - `supported_hold_position = 12.0`
  - `short_hold_seed = 20.0`
- pre/post-release shaping increased:
  - `pre_release_grasp = 140.0`
  - `post_release_grasp = 300.0`
  - `post_release_survival = 1320.0`

Result:

- reward: `28121.324 -> 28642.855`
- `contact_duration_sec`: `9.65 -> 9.73`
- `joint_palm_contact`: `192.82 -> 194.50`
- `ring_wrap_contact`: `193.79 -> 193.95`
- `palm_contact`: `221.29 -> 222.74`
- still below the final target; `drop` remained `1.0`

Current best checkpoint to continue from:

- `sim_rl/mujoco_playground/logs/AeroCanGraspV2Force-20260424-030353-CAN54_release_gate_tighten/checkpoints/000000655360`

Current conclusion:

- the policy now reliably forms the intended human-like wrap shape
- the bottleneck is no longer "cannot contact" but "cannot retain after release long enough"
- next iterations should keep DR off and focus on post-release retention/slip prevention until unsupported hold is comfortably above 10s, then add DR

## CAN58 `cradle_handover_lock`

Changes:

- fixed a release-gate mismatch so `min_release_active_fingers=4` is now actually enforced in the can env release check
- moved back toward the stronger `CAN55` support handover settings, then added a new `cradle_lock` reward
- increased post-release penalties and retention terms to bias the hand toward palm + middle/ring joint clamp after support exits

Result:

- reward: `34794.953 -> 34503.727`
- `contact_duration_sec`: `10.99 -> 11.06`
- `joint_palm_contact`: `219.83 -> 221.07`
- `ring_wrap_contact`: `218.47 -> 220.27`
- `slip_event`: `0.074 -> 0.238`
- `cradle_lock` stayed at `0.0`

Interpretation:

- the stronger release gate and handover improved wrap integrity a little
- but the first version of `cradle_lock` was too hard and never activated
- this branch confirmed the shape was right, but it also showed we still needed a softer post-release under-support signal

## CAN59 `soft_cradle_sliplock`

Changes:

- rewrote `cradle_lock` as a softer signal:
  - thumb support can come from force + rough under-position instead of a single hard geometric condition
  - reduced hard-zero gating and let palm/joint/ulnar signals contribute continuously
- increased `post_release_slip` and `post_release_pose_hold` so the new cradle signal is tied more tightly to actual velocity suppression

Result:

- reward: `34547.992 -> 34829.891`
- `contact_duration_sec`: `11.07 -> 11.05`
- `slip_event`: `0.230 -> 0.121`
- `support_released`: `44.38 -> 48.33`
- `cradle_lock`: `555.46 -> 574.47`
- `post_release_joint_palm_hold`: `1040.96 -> 1077.54`
- `post_release_survival`: `7415.45 -> 8036.87`

Interpretation:

- this was the cleanest improvement branch in the latest round
- the new cradle reward finally activated and helped reduce slip after support release
- unsupported retention shape improved, but duration is still stuck around `11s`, so the next issue is no longer learning the right grasp shape but turning that shape into longer hold time

## CAN60 `release_exposure`

Changes:

- moved support release slightly earlier again:
  - `release_after_sec = 9.8`
  - `release_ramp_sec = 1.6`
  - `force_release_after_sec = 13.2`
  - `min_release_force = 0.12`
- reduced support-phase reward more and increased post-release survival / cradle weighting
- used the stronger `CAN59` policy as the new base to increase honest unsupported practice

Result:

- reward: `33314.695 -> 33568.039`
- `contact_duration_sec`: `10.63 -> 10.66`
- `slip_event`: `0.109 -> 0.086`
- `support_released`: `45.82 -> 47.52`
- `cradle_lock`: `671.96 -> 764.33`
- `post_release_joint_palm_hold`: `1239.95 -> 1410.11`
- `post_release_survival`: `8949.44 -> 9627.42`

Interpretation:

- this branch is more honest about unsupported handover and much stronger on anti-slip shaping
- but it gives up too much hold time compared with `CAN59`
- so `CAN60` is useful as a reference for release honesty, while `CAN59` remains the better mainline checkpoint to continue from if the goal is pushing duration upward

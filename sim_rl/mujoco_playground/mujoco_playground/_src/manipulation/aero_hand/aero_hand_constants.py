# Copyright 2025 TetherIA Inc.
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Constants for TetherIA Aero Hand Open."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "aero_hand"
CUBE_XML = ROOT_PATH / "xmls" / "scene_mjx_cube.xml"
GRASP_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp.xml"
GRASP_HW6_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp_hw6.xml"
GRASP_HW6_PALMUP_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp_hw6_palmup.xml"

NQ = 16
NV = 16
NU = 7
NU_HW6 = 6

JOINT_NAMES = [
    # index
    "right_index_mcp_flex",
    "right_index_pip",
    "right_index_dip",
    # middle
    "right_middle_mcp_flex",
    "right_middle_pip",
    "right_middle_dip",
    # ring
    "right_ring_mcp_flex",
    "right_ring_pip",
    "right_ring_dip",
    # pinky
    "right_pinky_mcp_flex",
    "right_pinky_pip",
    "right_pinky_dip",
    # thumb
    "right_thumb_cmc_abd",
    "right_thumb_cmc_flex",
    "right_thumb_mcp",
    "right_thumb_ip",
]

ACTUATOR_NAMES = [
    # index
    "right_index_A_tendon",
    # middle
    "right_middle_A_tendon",
    # ring
    "right_ring_A_tendon",
    # pinky
    "right_pinky_A_tendon",
    # thumb
    "right_thumb_A_cmc_abd",
    "right_th1_A_tendon",
    "right_th2_A_tendon",
]

FINGERTIP_NAMES = [
    "if_tip",
    "mf_tip",
    "rf_tip",
    "pf_tip",
    "th_tip",
]


SENSOR_TENDON_NAMES = [
    "len_if",
    "len_mf",
    "len_rf",
    "len_pf",
    "len_th1",
    "len_th2",
]

SENSOR_JOINT_NAMES = [
    "len_th_abd",
]

SENSOR_HW6_POS_NAMES = [
    "hw_pos_thumb_rot",
    "hw_pos_thumb_flex",
    "hw_pos_index",
    "hw_pos_middle",
    "hw_pos_ring",
    "hw_pos_pinky",
]

_HW6_FORCE_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
SENSOR_HW6_FORCE_NAMES = [
    f"hw_tip_frc_{finger}_{taxel:02d}"
    for finger in _HW6_FORCE_FINGER_ORDER
    for taxel in range(16)
]

# ── V2 灵犀手 ──────────────────────────────────────────────────────────────

GRASP_V2_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp_v2.xml"
GRASP_V2_COACD_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp_v2_coacd.xml"
GRASP_V2_COACD_QBR_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp_v2_coacd_qbr.xml"
GRASP_V2_BOTTLE_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp_bottle_550ml.xml"
GRASP_V2_CAN_XML = ROOT_PATH / "xmls" / "scene_mjx_grasp_can_330ml.xml"

V2_NQ = 11
V2_NV = 11
V2_NU = 6

V2_JOINT_NAMES = [
    # index
    "right_index_mcp",
    "right_index_pip",
    # middle
    "right_middle_mcp",
    "right_middle_pip",
    # ring
    "right_ring_mcp",
    "right_ring_pip",
    # pinky
    "right_pinky_mcp",
    "right_pinky_pip",
    # thumb
    "right_thumb_cmc_abd",
    "right_thumb_cmc_flex",
    "right_thumb_mcp",
]

V2_ACTUATOR_NAMES = [
    "hw_thumb_rot",
    "hw_thumb_flex",
    "hw_index",
    "hw_middle",
    "hw_ring",
    "hw_pinky",
]

# V2 fingertip site 名称（传感器用，与 V1 相同）
V2_FINGERTIP_NAMES = FINGERTIP_NAMES

# V2 每指所有碰撞 geom 名称（efc_force 接触力提取用）
# 抓握时接触可能发生在指尖(tip)、远端(dist)或近端(prox)任意碰撞体
V2_FINGER_CONTACT_GEOMS = {
    "index":  ["if_prox_col", "if_dist_col", "if_tip_back_col", "if_tip_col"],
    "middle": ["mf_prox_col", "mf_dist_col", "mf_tip_back_col", "mf_tip_col"],
    "ring":   ["rf_prox_col", "rf_dist_col", "rf_tip_back_col", "rf_tip_col"],
    "pinky":  ["pf_prox_col", "pf_dist_col", "pf_tip_back_col", "pf_tip_col"],
    "thumb":  [
        "th_base_col",
        "th_mid_col",
        "th_tip_back_col",
        "th_tip_root_col",
        "th_tip_deep_col",
        "th_tip_col_1",
        "th_tip_col_2",
    ],
}

# V2 每指指尖/指腹碰撞 geom 名称（奖励、触觉观测、释放门控、诊断用）
# 近端/远端碰撞盒仍参与物理碰撞，但不能冒充三指指尖捏握接触。
V2_FINGERTIP_CONTACT_GEOMS = {
    "index":  ["if_tip_col"],
    "middle": ["mf_tip_col"],
    "ring":   ["rf_tip_col"],
    "pinky":  ["pf_tip_col"],
    "thumb":  ["th_tip_col_1", "th_tip_col_2"],
}

V2_SENSOR_POS_NAMES = SENSOR_HW6_POS_NAMES

V2_SENSOR_FORCE_NAMES = SENSOR_HW6_FORCE_NAMES

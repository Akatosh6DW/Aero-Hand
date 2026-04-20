import mujoco
import mujoco.viewer
from mujoco_playground._src.manipulation.aero_hand.grasp_cube import CubeGrasp

print("正在使用训练环境的高级逻辑加载物理法则...")
env = CubeGrasp()

print("提取物理模型并生成初始状态...")
# 获取系统编译好的完美模型
model = env.mj_model
# 根据模型，手动生成一份初始的物理世界数据
data = mujoco.MjData(model)

print("加载成功！正在召唤上帝视角...")
# 启动渲染器
mujoco.viewer.launch(model, data)

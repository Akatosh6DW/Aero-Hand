import mujoco
import mujoco.viewer
import os

# =======================================================
# 👇 请把你要看的 XML 文件的绝对路径粘贴在双引号里面 👇
# =======================================================
XML_PATH = "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_hw6.xml"
# 仅用于可视化排查：将重力临时置零，方便检查初始相对位置。
ZERO_GRAVITY_FOR_VIEW = False
# 将执行器目标自动对齐到当前姿态，避免启动后瞬间弹动。
AUTO_EQUILIBRATE_CTRL = True
# =======================================================

if not os.path.exists(XML_PATH):
    print(f"❌ 找不到文件，请检查上面的路径：\n{XML_PATH}")
else:
    # 自动提取文件夹路径和文件名
    xml_dir = os.path.dirname(XML_PATH)
    xml_name = os.path.basename(XML_PATH)

    # 【核心操作】强行把程序的工作目录切换到 XML 所在的文件夹
    # 这样 MuJoCo 就能以这个文件夹为起点去找 STL 3D 模型了
    os.chdir(xml_dir)

    print(f"⏳ 正在加载世界法则: {xml_name} ...")
    try:
        # 读取模型并生成初始状态
        model = mujoco.MjModel.from_xml_path(xml_name)
        if ZERO_GRAVITY_FOR_VIEW:
            model.opt.gravity[:] = 0.0
            print("🔧 已启用零重力可视化模式: gravity =", model.opt.gravity)
        data = mujoco.MjData(model)

        # 如果有 home keyframe，强制使用它，这样可视化姿态与训练 reset 对齐
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            nq, nu = model.nq, model.nu
            data.qpos[:] = model.key_qpos[key_id * nq : (key_id + 1) * nq]
            if nu > 0:
                data.ctrl[:] = model.key_ctrl[key_id * nu : (key_id + 1) * nu]
            if model.nmocap > 0:
                data.mocap_pos[:] = model.key_mpos[key_id * model.nmocap : (key_id + 1) * model.nmocap]
                data.mocap_quat[:] = model.key_mquat[key_id * model.nmocap : (key_id + 1) * model.nmocap]
        mujoco.mj_forward(model, data)

        if AUTO_EQUILIBRATE_CTRL and model.nu > 0:
            # 让每个执行器目标等于“当前实际值”，减少强制回位造成的瞬态弹动。
            for i in range(model.nu):
                trn_type = int(model.actuator_trntype[i])
                trn_id = int(model.actuator_trnid[i, 0])
                if trn_type == 0 and trn_id >= 0:  # joint actuator
                    qadr = model.jnt_qposadr[trn_id]
                    data.ctrl[i] = data.qpos[qadr]
                elif trn_type == 3 and trn_id >= 0:  # tendon actuator
                    data.ctrl[i] = data.ten_length[trn_id]
            mujoco.mj_forward(model, data)
            print("🔧 已启用控制目标平衡模式，初始状态应保持静态")

        # 打印手心与方块相对位置，便于快速判断是否在可抓取范围
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if site_id >= 0 and body_id >= 0:
            grasp_site = data.site_xpos[site_id].copy()
            cube_pos = data.xpos[body_id].copy()
            print("grasp_site:", grasp_site)
            print("cube_pos  :", cube_pos)
            print("cube-grasp:", cube_pos - grasp_site)
        
        print("✅ 加载成功！弹窗已生成 (按空格键可以播放/暂停时间)")
        
        # 召唤上帝视角！
        mujoco.viewer.launch(model, data)
        
    except Exception as e:
        print(f"\n❌ MuJoCo 引擎报错了: {e}")
        print("💡 提示：如果依然报错说找不到 'assets/xxx.STL'，说明模型文件不在这个 XML 的旁边！")
        print("请打开 XML 文件，搜索 <compiler meshdir=... />，看看它指向的路径对不对。")
# Sim and RL

The **TetherIA Aero Hand Open** features a physically accurate **tendon-driven actuation system** that emulates real cable mechanics, allowing control and observation directly in **tendon space**. This design provides a faithful bridge between simulation and real-world deployment.

The hand can be simulated across multiple physics platforms. Currently, it supports the **[MuJoCo simulator](https://mujoco.org/)**, where it is fully integrated into **[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)** for seamless visualization, training, and evaluation of tendon-space control policies. Support for additional simulation platforms is currently in progress.

- **Hand Model:** [MuJoCo Menagerie – TetherIA Aero Hand Open](https://github.com/TetherIA/mujoco_menagerie/tree/main/tetheria_aero_hand_open)  
- **Training Tools:** [MuJoCo Playground](https://github.com/TetherIA/mujoco_playground_pr)

---

## 🚀 Quick Start

If you’d like to **train your own policy**, follow these steps:

1. Install **MuJoCo Playground** following the [installation guide](https://github.com/TetherIA/mujoco_playground_pr/blob/main/README.md).  
2. Try the built-in **Z-axis rotation environment** for a quick demo, or create your own custom task following [this example](https://github.com/TetherIA/mujoco_playground_pr/blob/main/mujoco_playground/_src/manipulation/aero_hand/rotate_z.py).  
3. Launch training with:

```bash
python learning/train_jax_ppo.py --env_name AeroCubeRotateZAxis
```
You can replace `AeroCubeRotateZAxis` with the name of your own environment. After training, a rollout video will be saved as `rollout0.mp4`.

If you'd like to explore TetherIA Aero Hand Open within the MuJoCo simulator, follow these steps:

1. Follow the [official MuJoCo installation guide](https://mujoco.readthedocs.io/en/latest/programming/#getting-started).
2. The hand model is hosted under our fork of the MuJoCo Menagerie:
   ```bash
      git clone git@github.com:TetherIA/mujoco_menagerie.git
   ```
3. Navigate to the hand’s directory and open the simulation scene:
   ```bash
      cd mujoco_menagerie/tetheria_aero_hand_open
      simulate scene_right.xml
   ```

**Main features include:**
- Physically consistent tendon routing and pulley mechanics.  
- Full tendon-space control and observation interface.  
- JAX-based PPO training integrated with MuJoCo Playground.  

An overview of the hand is shown below:

| ![Rock](imgs/rock.png) | ![Paper](imgs/paper.png) | ![Scissor](imgs/scissor.png) |
|------------------------|------------------------|------------------------|


---

## 1. TetherIA Aero Hand Model

### 1.1 Modeling

The simulation model is derived from URDF models, accurately representing the real mechanical structure. The simulation reproduces the cable actuation through three essential components:

#### 1.1.1 Tendon Drives
Spatial tendons drive the fingers and thumb, following the same routing paths as the real hand.

#### 1.1.2 Springs
Modeled as tendon elements, springs generate restoring forces for finger extension, mirroring the real mechanical springs.

#### 1.1.3 Pulleys
Cylindrical pulleys guide cables and springs to ensure accurate routing and consistent actuation behavior.

| Front view | Close-up of index |
|-------------|------------------|
| ![skeleton](imgs/skeleton.png) | ![index](imgs/index_close_up.png) |

---

### 1.2 Parameters

#### 1.2.1 Mechanical Parameters
- **Joint limits, mass, and inertia:** directly taken from the URDF for physical fidelity.  
- **Pulley placement:** matches the real hand’s geometry, ensuring correct cable and spring routing.  
- **Validation:** the simulated tendon range (0.0459454 m) closely matches the real hand (0.04553 m) without manual adjustment.

#### 1.2.2 Tendon and Spring Specifications
- **Tendon properties:** identical to the real system.  
- **Spring properties:** match hardware specifications, except for the DIP spring—adjusted slightly for more realistic motion.

#### 1.2.3 Control Parameters
All remaining control parameters are fine-tuned to ensure consistent dynamics between simulation and hardware, including:
- Joint damping values  
- Actuator gains  
- Joint-specific damping coefficients. 



---

## 2. Training a Tendon-Space Policy

We introduce a **Z-axis rotation task** for the **TetherIA Aero Hand Open**, optimized using the following reward function:

reward = 1.0 × angular velocity - 1.0 × action rate + termination (-100.0)


The optimization variables include **tendon lengths** and the **thumb abduction joint**, both corresponding to the real actuation system. This design ensures that the same control and sensing channels can be directly applied in **sim-to-real deployment**.

To train the policy:

```bash
python learning/train_jax_ppo.py --env_name AeroCubeRotateZAxis
```

While training results vary slightly due to stochasticity, runs consistently **converge toward a positive total reward**. Below are example reward curves obtained using PPO:

**Overall Reward**
![overall](imgs/reward_overall.png)

**Angular Velocity Reward**
![reward_angvel](imgs/reward_angvel.png)

**Action-Rate Penalty**
![penalty_action_rate](imgs/penalty_action_rate.png)

**Termination Penalty**
![penalty_termination](imgs/penalty_termination.png)

---

## 3. Running a Pretrained Policy

This example builds upon **[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)** as an educational platform. We are actively working to merge **[our implementation](https://github.com/TetherIA/mujoco_playground_pr)** into the official repository.

To test a trained policy:
```bash
python learning/train_jax_ppo.py
--env_name AeroCubeRotateZAxis
--play_only
--load_checkpoint_path path/to/checkpoints
```

This will:
- Load the trained PPO policy  
- Run rollouts in simulation  
- Visualize the manipulation behavior

| Seed 0 | Seed 1 | Seed 14 |
|---------|---------|---------|
| ![seed0](imgs/cube_rotation.gif) | ![seed1](imgs/cube_rotation1.gif) | ![seed14](imgs/cube_rotation14.gif) |

**Sim-to-Real Deployment on the TetherIA Aero Hand Open**

![Z-Axis Rotation Demo](imgs/output.gif)

*Note: The finger markers are for visualization only and are **not** used for control.*

---

## 🧩 Acknowledgements

Dependency:  
- [**MuJoCo Playground**](https://github.com/google-deepmind/mujoco_playground). 

<div align="center">

Made with ❤️ by **TetherIA Robotics**

</div>

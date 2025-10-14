# Aero Hand Open

<div style={{ display: "flex", justifyContent: "center", gap: "1rem", marginTop: "1rem" }}>
  <div style={{ textAlign: "center" }}>
    <img src="/img/banner.jpg" alt="Open Palm Pose" width="700"/>
    <p><em>Aero Hand Open</em></p>
  </div>
</div>

**Aero Hand Open** is a tendon-driven, under-actuated dexterous robotic hand built for research in manipulation, reinforcement learning, and embodied AI.  
It is designed for reproducibility, transparency, and ease of integration across hardware, firmware, and software layers.

---

## Hardware Overview

The hand adopts a **tendon-driven actuation architecture**, where each motor drives multiple joints through cable routing.  
This approach achieves high dexterity with fewer actuators, resulting in a lightweight and compliant design.

**Key characteristics:**
- 15 Degrees of Freedom (7 active, 8 passive)
- 3 active DoFs for the thumb, enabling real dexterity
- One-motor-per-finger tendon actuation
- Passive 1:1 coupling between DIP and PIP joints
- Adaptive coupling at MCP joint for compliant contact behavior
- Compact servo actuator modules with integrated encoders
- Modular 3D-printable mechanical components
- 6 V DC power input and USB communication
- Fully open-source CAD and PCB design

All mechanical and electrical design files are available in the [`aero-open-hardware`](https://github.com/TetherIA/aero-open-hardware) repository.

---

## Firmware and Control

The onboard controller runs custom firmware supporting:
- Position, velocity, and torque control modes
- High-frequency communication over USB
- Real-time command streaming via ROS 2 and Python SDK
- Built-in calibration and diagnostics utilities

Firmware source code is hosted in [`aero-open-firmware`](https://github.com/TetherIA/aero-open-firmware).

---

## SDK and ROS 2 Integration

Developers can interface with the hand through:
- A **Python SDK** for joint, tendon, and sensor interfaces
- **ROS 2 packages** exposing scripting, teleoperation, and logging
- Example nodes for control, data collection, and policy playback

Software API is maintained in [`aero-open-sdk`](https://github.com/TetherIA/aero-open-sdk).

---

## Simulation and Learning

Aero Hand Open is fully supported in **MuJoCo** with upcoming Issac Sim support, with tendon-level actuation and observation.  
It integrates with popular RL and imitation learning frameworks, including **MuJoCo Playground** and **LeRobot** (coming soon), enabling a consistent sim-to-real workflow.

Use simulation to train control policies, benchmark algorithms, and prototype manipulation tasks before deployment on real hardware.

---

## Repository Structure

| Component | Repository | Description |
|------------|-------------|-------------|
| Hardware | [`aero-open-hardware`](https://github.com/TetherIA/aero-open-hardware) | CAD models, assembly, and tendon routing |
| Firmware | [`aero-open-firmware`](https://github.com/TetherIA/aero-open-firmware) | Embedded control software |
| SDK / GUI | [`aero-open-sdk`](https://github.com/TetherIA/aero-open-sdk) | Python SDK and GUI |
| Documentation | [`aero-open-docs`](https://github.com/TetherIA/aero-open-docs) | Documentation source |
| Monorepo | [`aero-hand-open`](https://github.com/TetherIA/aero-hand-open) | Consolidated repository for community contributions |

---

## Getting Started

- [Build and assemble the hand →](./assembly.md)  
- [Flash and verify firmware →](./firmware.md)  
- [Connect via ROS 2 or Python SDK →](./sdk.md)
- [Getting Started with GUI of SDK →](./getting_started.md)  
- [Run simulation in MuJoCo →](./hand_sim.md)

---

## License

All components of Aero Hand Open are released under permissive open-source licenses.  
Please refer to individual repositories for license details.

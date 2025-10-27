**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) © 2025 TetherIA


# Aero Hand Hardware

## Hardware Overview

The hand adopts a **tendon-driven actuation architecture**, where each motor drives multiple joints through tendon routing.  
This approach achieves high dexterity with fewer actuators, resulting in a lightweight and compliant design.

**Key characteristics:**
- 16 Degrees of Freedom (7 active, 9 passive)
- 3 active DoFs for the thumb, enabling real dexterity
- One-motor-per-finger tendon actuation
- Passive 1:1 coupling between DIP and PIP joints
- Adaptive coupling at MCP joint for compliant contact behavior
- Compact servo actuator modules with integrated encoders
- Modular 3D-printable mechanical components

## Specifications

**Physical:**
- Weight: 389 grams
- Size: 198 mm x 95 mm x 53.5 mm
- Material: PLA

**Actuation:**
- 7 Coreless Serial Bus Servos, tendon driven, backdriveable
- Finger tip force: ~10 N
- Open+close speed: ~1.2 Hz per full cycle

**Joint Configuration:**
- 7 active DoF, 16 joints (thumb x3, fingers x1 underactuated)
- Joint range of motion:
  - Finger joints: 90°
  - Thumb abduction: 0°-100°
  - Thumb CMC flex: 55°
  - Thumb proximal/distal: 90°

**Electronics:**
- Communication interface: USB 2.0
- MCU: ESP32-S3
- Working voltage: 6 V
- Maximum current: 8 A


## CAD 
- CAD files are provided in step format for the right and left hand.
- We have also uploaded all CAD to [OnShape](https://cad.onshape.com/documents/afc7e0ca7eb6d412ec8771f8/w/bc4d7e45e17e23d622d2bad2/e/1e3862db4d4df7d9dca6f286?renderMode=0&uiState=68e5abd64106f26dc459da44). There is a motion study in the OnShape workspace, please do not refer to this CAD to build you hand, it is **NOT** up to date.
- A one click print file is provided for both right and left hands, the print orientations have been optimized to minimize supports and we reccomend keeping them the same. Printed on Bambu Labs X1C with 0.4mm nozzle, default print settings for 0.20mm standard layer height except: enable tree supports and supports only on build plate.
- Silicone mold files are also provided as step files and one click print files.

## Assembly
- Please refer to the [Assembly Instructions pdf](https://docs.tetheria.ai/docs/assembly) to assemble the hand, we are always working to improve the assembly instructions and appreciate any feedback. A video assembly will be released soon.
- The [BOM](Assembly/BOM.csv) is provided and a list of tools to help with assembly is also provided.
- Silicone mold instructions are provided, a TPU alternative will also be available coming soon


## PCB
We designed two custom PCBs for the Aero Hand Open:

**PCB Board A (Hand PCB):**
- Fits inside the hand.
- Features 8 Molex 3-pin connectors for attaching servos.
- Includes 1 JST 3-pin (P-3.96mm) connector that exits the hand, connecting to a long cable.
- The long cable is used to connect to PCB B.

**PCB Board B:**
- Connects to PCB A via the long cable.
- Used for further signal routing, sending signals from PC and power management (details in project files).

All relevant files are provided in the [PCB](./PCB) folder:
- Bill of Materials (BOM) [Board A](./PCB/Aero_hand_open_boarda/PCB%20BOM%20-%20BoardA_BOM.csv), [Board B](./PCB/Aero_hand_open_boardb/PCB%20BOM%20-%20BoardB_BOM.csv)
- Component Placement List (CPL)
- STEP files for mechanical integration
- Gerber ZIP file for PCB manufacturing
- Schematic and layout files
- Project folder with all design resources
- KiCad project files

Please refer to the files in the [PCB](./PCB) folder for manufacturing, assembly, and integration details. Components folder includes all the components used for the project.


## Powering up the hand

Referring to our docs in https://docs.tetheria.ai/docs/hardware_setup




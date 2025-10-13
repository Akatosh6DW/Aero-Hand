# Aero Hand Open Firmware

This repository contains the firmware code for the **Aero Hand**, designed for control, homing, calibration, and telemetry of the TetherIA Aero Hand prosthetic.  
The firmware runs on an **ESP32-S3 (Seeed Studio XIAO ESP32S3)** and communicates with a PC host application via USB serial.

---

## 🔌 Communication Protocol

The firmware uses a **fixed 16-byte frame** for all communication between PC and firmware.

### Frame Structure (TX and RX)
| Bytes     | Field            | Description                                                                 |
|-----------|-----------------|-----------------------------------------------------------------------------|
| 0         | **Opcode**      | Command / response code (e.g., `0x01` for HOMING, `0x04` for TRIM).         |
| 1         | **Filler**      | Always `0x00` (reserved for future use).                                    |
| 2..15     | **Payload**     | 14-byte payload. May contain parameters (channels, degrees, IDs, etc.) or be all zeros in acknowledgments. |

- **All commands from PC → firmware are 16 bytes.**  
- **All responses (ACKs) from firmware → PC are also 16 bytes.**

---

## 🚀 Supported Modes

### 1. HOMING (`0x01`)
- **PC → Firmware:**  
  - Opcode `0x01`  
  - Payload: all zeros  
- **Firmware action:**  
  - Runs the **full homing sequence** for all servos (thumb & fingers).  
  - Each joint is driven until the servo **hits its current limit**; this is used to determine the **extend limit**.  
  - The firmware calibrates servo motion spans based on measured **extend_count** and **grasp_count**.  
  - **Blocking call:** during homing, all other commands are ignored until the procedure completes and an ACK is returned under a given timeout.  
- **Firmware → PC (ACK):**  
  - `[0x01, 0x00, 14×0x00]`  
  - Indicates homing complete.

---

### 2. SET_ID (`0x02`)
- **PC → Firmware:**  
  - Opcode `0x02`  
  - Payload:  
    - Word 0 (2 bytes): **new ID** (recommended range: 0–6, since 7 servos)  
    - Word 1 (2 bytes): **current limit** (0–1023)  
- **Firmware action:**  
  - Scans the servo bus for the first connected servo.  
  - Updates its EEPROM with the new ID and the current limit.  
- **Firmware → PC (ACK):**  
  - `[0x02, 0x00, oldId(2B), newId(2B), currentLimit(2B), rest zeros]`  
- **Notes:**  
  - IDs must be unique and in the range **0–6**.  
  - Recommended current limit = **1023 (maximum)** to enable strong grasping power and full servo capability.


---

### 3. TRIM (`0x03`)
- **PC → Firmware:**  
  - Opcode `0x03`  
  - Payload:  
    - Word 0 (2 bytes): Servo channel (0–6)  
    - Word 1 (2 bytes): Degrees offset (+/- 360°)  
- **Firmware action:**  
  - Adjusts the stored **extend_count** (fully open position) for the specified channel.  
  - Saves the new extend count in **NVS**, so the calibration is **persistent across reboots**.  
- **Firmware → PC (ACK):**  
  - `[0x03, 0x00, channel(2B), extendCount(2B), rest zeros]`  
- **Channel Mapping:**  
  - 0 → Thumb abduction  
  - 1 → Thumb flexion  
  - 2 → Thumb tendon  
  - 3 → Index finger  
  - 4 → Middle finger  
  - 5 → Ring finger  
  - 6 → Pinky finger  
- **Usage notes:**  
  - Use TRIM for **fine-tuning the open (extend) position**.  
  - Avoid large offsets. If you see misalignment after trimming negative (e.g., –50), you can reverse-tune with a positive trim (e.g., +50). Generally, values of +10 or -10 are recommended.
  - To restore default calibration, run **HOMING**, which resets values to baseline.

---


### 5. CTRL_POS (`0x11`)
- **PC → Firmware:**  
  - Opcode `0x11`  
  - Payload: 7 words (2 bytes each), one for each channel [0..6].  
  - Range: `0–65535` mapped linearly to servo’s calibrated `extend_count` → `grasp_count`.  
- **Firmware action:**  
  - Uses **SyncWritePosEx** to command all 7 servos simultaneously.  
- **Firmware → PC (ACK):**  
  - None (movement commands are not acknowledged).  

---



### 6. CTRL_TOR (`0x12`)
- **PC → Firmware:**  
  - Opcode `0x12`  
  - To be Implemented
---
### 6. Telemetry / GET Modes
The firmware supports batched telemetry reads via **SyncRead**.

- **GET_POS (`0x22`)** → Responds with `[pos0..pos6]` (7×u16).
- **GET_VEL (`0x23`)** → Responds with `[vel0..vel6]` (7×u16).  
- **GET_CURR (`0x24`)** → Responds with `[cur0..cur6]` (7×u16).  
- **GET_TEMP (`0x25`)** → Responds with `[tmp0..tmp6]` (7×u16).  

Each telemetry response is also a **16-byte frame**.

---

## 📜 Example Sequences

### Homing from PC
1. PC sends: `[0x01, 0x00, 14×0x00]`  
2. Firmware homes all servos.  
3. Firmware replies: `[0x01, 0x00, 14×0x00]`  

### Trim channel 3 by –100°
1. PC sends: `[0x04, 0x00, 0x03,0x00, 0x9C,0xFF, 10×0x00]`  
2. Firmware updates extend count, saves to NVS.  
3. Firmware replies: `[0x04, 0x00, 0x03,0x00, ext_lo,ext_hi, rest zeros]`

---

## 🧩 Implementation Notes
- **Bus locking:**  
  All sync read/write operations use `gBusMux` semaphore to prevent collisions.  
- **Persistence:**  
  Extend counts modified via TRIM are stored in **NVS** (`prefs`) for persistence across reboots.  
- **Timing:**  
  Homing sequence may take up to **25 seconds per channel**. During this time, **all other commands are ignored**.  
- **Frame alignment:**  
  Unknown or invalid opcodes are consumed but ignored, to maintain framing alignment.

---

## ✅ Summary
- Communication is always **16-byte aligned**.  
- Homing (`0x01`), Set ID (`0x03`), and Trim (`0x04`) **send back ACKs**.  
- Position control (`0x11`) is **one-way command**.
- Telemetry Commands -GET_POS (`0x22`), GET_VEL (`0x23`), GET_CURR (`0x24`), GET_TEMP (`0x25`) are **data receiving commands**.
- Extend calibration done using Trim Servo survives power cycles via NVS storage but if you call homing in between it takes the baseline values.  
- Firmware is **robust to stale/invalid frames** by ignoring them safely.  
- **Trim adjustments persist across reboots**. Use **HOMING** to reset calibration.  
- GUI **Zero All** = shortcut for sending `CTRL_POS` with all channels → extend.  

---

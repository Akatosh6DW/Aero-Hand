# Firmware

This guide covers architecture, build options, and a detailed communication protocol specification used by the **ESP32‑S3 (Seeed Studio XIAO ESP32S3)** to drive **Feetech HLS3606M servos**.

---

## 🧭 Overview

The firmware exposes a compact **fixed 16‑byte binary serial protocol** for commanding seven actuators (thumb & fingers) and for querying telemetry (position, velocity, current, temperature). It also implements **homing**, **set-id**, **persistent trims**, and safety behaviors on‑board, so a host PC can remain simple.

**Channels (7):**

1. Thumb CMC Abduction Actuator
2. Thumb CMC Flexion Actuator
3. Thumb Tendon Actuator
4. Index Finger Actuator 
5. Middle Finger Actuator
6. Ring Finger Actuator
7. Pinky Finger Actuator

---

## 🧰 Hardware & Software Requirements

* **MCU:** Seeed Studio XIAO ESP32‑S3 (8 MB flash recommended)
* **Servos:** Feetech HLS/SC family (e.g., HLS3606M), IDs mapped to 7 actuators
* **Power:** Servo rail 6–9 V with adequate current capacity; USB-C 5 V for MCU. We recommend a stable power supply at 6V for driving the hand capable of delivering upto 10A.
* **Tools:** Arduino IDE **or** PlatformIO

---

## ⚡ Quick Start

1. **Select the hand** (left/right) in `HandConfig.h` or via a build flag (see below).
2. **Build & flash:**
   * **PlatformIO:** choose `seeed_xiao_esp32s3` ➜ **Upload**
   * **Arduino IDE:** open main sketch ➜ set board to **XIAO ESP32S3** ➜ **Upload**
3. **First boot:** open a serial monitor at the configured baud (commonly 921600 or 1 000 000). Verify the board enumerates and servos respond OK if power is connected.

---

## 📁 Repository Layout

```
/firmware
  ├─ firmware_v0.1.0.ino         # Main sketch: init, parser, handlers, tasks
  ├─ HandConfig.h                # LEFT_HAND / RIGHT_HAND selection
  ├─ Homing.h                    # Homing API + ServoData type
  ├─ Homing.cpp                  # Homing implementation + baselines
  └─ libraries/                  # Third‑party libraries as needed
```

---

## 🔁 Build Configuration: Left vs Right Hand

You can switch hands either by editing `HandConfig.h` **or** using build flags.

* **PlatformIO** (`platformio.ini`):

  ```ini
  [env:seeed_xiao_esp32s3]
  platform = espressif32
  board = seeed_xiao_esp32s3
  framework = arduino
  build_flags = -DRIGHT_HAND       ; or -DLEFT_HAND
  ```
* **Arduino IDE:** temporarily uncomment the macro in `HandConfig.h`.

> Tip: Keep a default committed (often `RIGHT_HAND`) and override via CI or local flags when building the opposite hand.
> Note: You can also use the aero-hand-gui to upload a bin file directly onto the ESP32S3 without using any of this software.

---

## 🔌 Communication Protocol

The protocol is **always 16 bytes** per frame, both **to** and **from** the device. Payload words are **little‑endian**.

### Frame Format

| Bytes | Field       | Description                                 |
| ----: | ----------- | ------------------------------------------- |
|     0 | **OPCODE**  | Command / request code                      |
|     1 | **FILLER**  | Always `0x00` (reserved)                    |
| 2..15 | **PAYLOAD** | 14‑byte payload; semantics depend on opcode |

> **ACK/Responses** also use the same 16‑byte frame shape; byte 0 contains the response opcode, byte 1 is `0x00`.

### Opcode Reference

> The table below is the canonical mapping used by this firmware version.

| Opcode | Name         | Direction | Payload (PC→Device unless noted)  | Notes                                                                                                             |
| :----: | ------------ | :-------: | --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `0x01` | **HOMING**   |   PC→Dev  | none (payload = zeros)            | Runs full homing & calibration; blocks other commands until complete; returns an ACK.                             |
| `0x02` | **SET_ID**   |   PC→Dev  | `u16 new_id`, `u16 current_limit` | Scans bus for the single attached servo, sets its ID and current limit; returns an ACK with old/new ID and limit. |
| `0x03` | **TRIM**     |   PC→Dev  | `u16 channel`, `s16 degrees_off`  | Adjusts stored **extend_count** for a channel; persists in NVS; returns an ACK with the resulting extend count.   |
| `0x11` | **CTRL_POS** |   PC→Dev  | 7×`u16` (ch0..ch6)                | Position write for all channels. 0..65535 spans each channel’s **extend ↔ grasp**. No ACK (fire‑and‑go).          |
| `0x12` | **CTRL_TOR** |   PC→Dev  | 7x`u16` (ch0..ch6)                | Torque write for all channels. 0..1000 is the limit for setting torque to each servo in direction **extend ↔ grasp** . No ACK (fire-and-go)                                                                     |
| `0x22` | **GET_POS**  |   PC→Dev  | none                              | Device replies with 7×`u16` raw positions.                                                                        |
| `0x23` | **GET_VEL**  |   PC→Dev  | none                              | Device replies with 7×`u16` velocities.                                                                           |
| `0x24` | **GET_CURR** |   PC→Dev  | none                              | Device replies with 7×`u16` currents.                                                                             |
| `0x25` | **GET_TEMP** |   PC→Dev  | none                              | Device replies with 7×`u16` temperatures.                                                                         |
| `0x31` | **SET_SPE**  |   PC→Dev  | `u16 id`, `u16 speed_limit`        | Sets the speed limit for the specified servo ID.                                                                  |
| `0x32` | **SET_TOR**  |   PC→Dev  | `u16 id`, `u16 torque_limit`       | Sets the maximum torque for the specified servo ID (different from torque control; this is a limit, not a command).|

#### Position Mapping (CTRL_POS)

For each channel *i*, the firmware maps host value `u16[i]∈[0,65535]` linearly to the servo’s raw count using the per‑channel `extend_count` (open) and `grasp_count` (closed), clamped to `[0,4095]`. Direction is handled via `servo_direction`.

#### Torque Control (CTRL_TOR)

For each channel, the firmware maps the incoming torque value in the range 0..1000 linearly to the raw torque sent to the servo, applying it only in the direction from extend to grasp.

#### Example Sequences

* **Homing**
  PC → Dev: `[0x01, 0x00, 14×0x00]`
  Dev homes all servos → Dev → PC: `[0x01, 0x00, 14×0x00]`

* **Trim channel 3 by −100°**
  PC → Dev: `[0x03, 0x00, 0x03,0x00, 0x9C,0xFF, 10×0x00]`
  Dev updates extend count & saves to NVS → Dev → PC: `[0x03, 0x00, 0x03,0x00, ext_lo,ext_hi, rest 0]`

> **Framing robustness:** Unrecognized opcodes are consumed/ignored to maintain alignment; malformed frames are dropped.

---

## 🧱 Firmware Architecture

### Data Structures

```c++
struct ServoData {              // one per channel
  uint16_t grasp_count;         // closed endpoint (0..4095)
  uint16_t extend_count;        // open endpoint  (0..4095)
  int8_t   servo_direction;     // +1 or −1
};
```

* **Baselines:** compile‑time constants for LEFT and RIGHT hands (7 entries each). A runtime working copy `sd[7]` is refreshed from the baseline (e.g., at boot or before homing).

### Homing Module

* `HOMING_start()` — runs the homing routine for all servos.
* `HOMING_isBusy()` — returns `true` while homing is in progress.
* `resetSdToBaseline()` — copies the correct baseline values into `sd[7]`.

### Concurrency & I/O

* **Bus locking:** all sync read/write operations acquire a semaphore (e.g., `gBusMux`) to prevent collisions.
* **Persistence:** trims (extend counts) are stored in **NVS** so they survive reboots; a full **HOMING** reset returns to baseline.
* **Telemetry:** batched `SyncRead` used for GET_*; CSV printing (if enabled) multiplexed through a single serial writer.

---

## 🧰 Homing Behavior

1. Load the appropriate baseline via `resetSdToBaseline()`.
2. For each servo, drive slowly toward a mechanical stop while monitoring current to detect contact.
3. Calibrate the offset at contact, back‑off/settle, then move to a consistent **extend** posture.

**Notes:** Homing may take around a minute or two overall since each servo is given a timeout of 25s to find a hard mechanical stop in a given direction. During homing, other commands are ignored (buffer on the host if needed), So it will wait for an ACK for a given time period.

---

## 🧩 Extending the Firmware (Add a New Command)

Follow this pattern to add features like LED blink, torque enable, or special saves.

**Step 1 — Define a new opcode**

```c++
static const uint8_t OP_BLINK = 0x42;   // choose an unused byte
```

**Step 2 — Implement a small handler**

```c++
static void handleBlink(uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    digitalWrite(LED_BUILTIN, HIGH); delay(80);
    digitalWrite(LED_BUILTIN, LOW);  delay(80);
  }
}
```

**Step 3 — Wire it into the parser**

```c++
case OP_BLINK: {
  uint8_t count = payload[0];   // interpret first payload byte
  handleBlink(count);
  break;
}
```

**Step 4 — (Optional) ACK**
Return a 16‑byte response if the host should be notified of completion or status.

**Step 5 — Update host tools**
Extend your Python SDK/GUI (e.g., `aero_hand.py`) to emit and consume the new opcode.

---

## 🔐 Safety & Best Practices

* Start with conservative speeds/torque when testing.
* Ensure the servo supply can deliver peak current without large drops.
* Avoid high‑rate position writes during homing.
* Use `HOMING` to re‑establish known geometry if trims become inconsistent.
* Always check power if the servos are not moving.

---

## 🧯 Troubleshooting

* **Wrong hand geometry:** Verify build flag (`-DLEFT_HAND` vs `-DRIGHT_HAND`).
* **Servos move opposite:** Check `servo_direction` or swap extend/grasp counts. We recommend not to change the servo_direction , extend_count can be changed by Trim Servo.
* **Homing stalls:** Inspect current limits/timeouts; verify mechanics move freely.
* **No serial activity:** Confirm COM port/baud and that frames are exactly 16 bytes with valid opcodes.

---

## 📄 License

This project is licensed under **Apache License‑2.0** 

---
## 🤝 Contribution

We welcome community contributions!

If you would like to improve the [Firmware](https://github.com/TetherIA/aero-open-firmware) or add new features:

1. Fork and create a feature branch.
2. Add or modify opcodes and handlers as described in Section 9.
3. Include brief unit/bench tests where possible.
4. Commit your changes with clear messages.
5. Push your branch to your fork.
6. Open a PR with a summary and scope of changes.

---

## 🤝 Support & Contact

* Open a GitHub Issue on the project repository
* Email: **[contact@tetheria.ai](mailto:contact@tetheria.ai)**

<div align="center">

Made with ❤️ by **TetherIA Robotics**

</div>

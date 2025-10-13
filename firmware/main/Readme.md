<p align="center">
  <img alt="Aero Hand Open by TetherIA" src="assets/logo.png" width="30%">
  <br/><br/>
</p>

# Aero Hand Open Firmware v0.1.0 — README

A reference guide for building, flashing, using, and extending the **Aero Hand Open** firmware that runs on an ESP32‑S3 (Seeed Studio XIAO ESP32S3) and drives Feetech smart servos - Feetech HLS3606M.

---

## Table of Contents
- [1) Overview](#1-overview)
- [2) Repository Layout](#2-repository-layout)
- [3) Hardware & Software Requirements](#3-hardware--software-requirements)
- [4) Quick Start](#4-quick-start)
- [5) Firmware Architecture](#5-firmware-architecture)
- [6) Selecting LEFT vs RIGHT Hand at Build Time](#6-selecting-left-vs-right-hand-at-build-time)
- [7) Serial Protocol](#7-serial-protocol)
- [8) Homing Behavior](#8-homing-behavior)
- [9) Extending the Firmware with a New Command](#9-extending-the-firmware-with-a-new-command)
- [10) Code Reference (quick lookup)](#10-code-reference-quick-lookup)
- [11) Safety & Best Practices](#11-safety--best-practices)
- [12) Troubleshooting](#12-troubleshooting)
- [13) Contribution](#13-contribution)
- [14) License](#14-license)
- [15) Support](#15-Support)

---

## 1) Overview

The firmware exposes a compact **fixed 16‑byte binary serial protocol** so a host PC can command finger positions and query telemetry while the device handles **homing**, calibration, and basic safety on‑board.

**Major components**
- **`firmware_v0.1.0.ino`** — Main sketch: hardware init, serial command parser, periodic tasks, and command handlers.
- **`HandConfig.h`** — **Defines LEFT_HAND or RIGHT_HAND** (build‑time) to select the correct baseline calibration and channel directions.
- **`Homing.h` / `Homing.cpp`** — Implements the **homing** routine (`HOMING_start()` / `HOMING_isBusy()`), plus baseline tables and helpers.

**Servos / channels** (7):
1. Thumb CMC Abduction
2. Thumb CMC Flexion
3. Thumb Tendon (thumb curl/tendon drive)
4. Index
5. Middle
6. Ring
7. Pinky

---

## 2) Repository Layout

```
/main
  ├─ firmware_v0.1.0.ino          # main sketch
  ├─ HandConfig.h                 # LEFT_HAND / RIGHT_HAND selection
  ├─ Homing.h                     # homing API + ServoData type
  ├─ Homing.cpp                   # homing implementation + baselines
  └─ (libraries as needed)
```

---

## 3) Hardware & Software 

- **MCU**: Seeed Studio XIAO ESP32‑S3 (8 MB flash recommended)
- **Servos**: Feetech HLS3606M (IDs mapped to the 7 joints)
- **Power**: Power Servo rail at 6 with adequate current (Max Current up to 10A) , separate from USB 5 V
- **Tools**: Arduino IDE **or** PlatformIO

---

## 4) Quick Start

### A. Select LEFT or RIGHT hand
In `HandConfig.h`, **define exactly one** of the following macros, or pass a build flag (see §6):
```c++
// #define LEFT_HAND
#define RIGHT_HAND
```

The choice selects the correct baseline calibration table at build time.

### B. Build & Flash
- **PlatformIO**: open the project, pick `seeed_xiao_esp32s3`, then **Upload**.
- **Arduino IDE**: open `firmware_v0.1.0.ino`, set Board to *XIAO ESP32S3*, then **Upload**.  
  **Note:** If you are using the Arduino IDE for the first time, you may need to install the ESP32 board support. Go to the Board Manager by pressing Ctrl+Shift+B, then search for "esp32". Find "esp32 by Espressif Systems" and install it. After installation, change the board under the Tools menu and select "XIAO_ESP32S3" from the list.

  <img width="1209" height="959" alt="image" src="https://github.com/user-attachments/assets/9305dd56-5df3-4f53-b719-54a90c7b7bcd" />

  <img width="463" height="283" alt="image" src="https://github.com/user-attachments/assets/48b62680-541a-425d-b0bc-8da7965821cb" />


### C. First Boot
1. Connect USB, open a serial monitor at the project baud rate (e.g., 1,000,000 or 115,200 as configured - default 921600).
2. On Pressing Reset, Using the Serial monitor make sure that you receive OK from all 7 Servos.

---

## 5) Firmware Architecture

### 5.1 Data structures
- **`ServoData`**: per‑channel configuration `{grasp_count, extend_count, servo_direction}` used to map 16‑bit host commands into per‑servo raw counts (0..4095).
- **Baselines**: compile‑time constants for LEFT and RIGHT hands (7 entries each). A runtime working copy `sd[7]` is initialized from whichever baseline the build selects.

### 5.2 Homing module
- **`HOMING_start()`**: runs the homing routine for all 7 servos.
- **`HOMING_isBusy()`**: returns true while homing is in progress.
- **`resetSdToBaseline()`**: copies the correct baseline into the working array before homing.

### 5.3 Main loop
The main loop performs:
- USB serial polling
- Fast‑path command decode (see §7)
- Optional periodic telemetry sampling and printing

---

## 6) Selecting LEFT vs RIGHT Hand at Build Time

You can switch hands either by editing `HandConfig.h` **or** using build flags.

- **PlatformIO** (`platformio.ini`):
  ```ini
  [env:seeed_xiao_esp32s3]
  platform = espressif32
  board = seeed_xiao_esp32s3
  framework = arduino
  build_flags = -DRIGHT_HAND      ; or -DLEFT_HAND
  ```

- **Arduino IDE**: temporarily uncomment the macro in `HandConfig.h`.

> **Tip:** Keep the default checked into version control (usually `RIGHT_HAND`) and apply overrides via build flags in CI or per‑developer settings.

---

## 7) Serial Protocol

### 7.1 Frame format (fixed 16 bytes)
- **Byte 0**: `OPCODE` (command / request)
- **Byte 1**: `0x00` (reserved filler)
- **Bytes 2..15**: 14‑byte payload (type depends on opcode)

### 7.2 Typical opcodes (examples)
- `0x01` — `HOMING` (no payload or optional params)
- `0x11` — `CTRL_POS` (write positions): payload is **seven little‑endian `uint16_t`** values, one per servo, representing 0..65535 spanning each channel’s configured **extend ↔ grasp** range.
- `0x22` — `GET_POS`, `0x23` — `GET_VEL`, `0x24` — `GET_CURR`, `0x25` — `GET_TEMP`

> The exact opcode set can evolve. Check the top of `firmware_v0.1.0.ino` for the authoritative constants used by your build.

### 7.3 Position mapping
For each channel *i*, the firmware maps a 16‑bit value `u16[i]` to the servo’s raw count using the per‑channel `extend_count` and `grasp_count` span, clamped to `0..4095`. Direction (sign) is handled via `servo_direction`.

---

## 8) Homing Behavior

The homing routine:
1. Loads the correct baseline via `resetSdToBaseline()`.
2. For each servo, **slowly drives towards a mechanical stop** while monitoring current (a “zero with current” approach).
3. Calibrates offset at contact, performs a small back‑off/settling motion, and finally moves to a defined **extend** posture for consistency across boots.

Timing, current limits, and per‑servo special cases (e.g., thumb tendon) are enforced internally to reduce stress on the mechanism.

> You can safely call `HOMING_start()` at any time from the serial command handler. While homing, ignore position writes or buffer them on the host side.

---

## 9) Extending the Firmware with a New Command

When you want to add a new feature (e.g., LED blink, torque enable/disable, save trim, etc.), follow this pattern:

### Step 1 — Add a new **control byte** (opcode)
At the top of `firmware_v0.1.0.ino`, define a unique `#define` or `static const uint8_t` value not already in use.
```c++
// Example: a simple LED or debug action
static const uint8_t OP_BLINK = 0x42;   // choose a free byte
```

### Step 2 — Implement a handler function
Keep the switch‑case light by moving logic into a small function.
```c++
static void handleBlink(uint8_t count) {
  for (uint8_t i = 0; i < count; ++i) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(80);
    digitalWrite(LED_BUILTIN, LOW);
    delay(80);
  }
}
```

### Step 3 — Add a `case` in the serial parser
Consume the payload as per use and call the function as per your handler.
```c++
case OP_BLINK: {
  uint8_t count = buf[1];                 // interpret first payload byte
  handleBlink(count);
  break;
}
```

### Step 4 — (Optional) Add a response
If the command should acknowledge or return data, write a 16‑byte response using the same framing convention (byte 0 = response opcode, byte 1 = 0x00, bytes 2..15 = payload).

### Step 5 — Update host tools
If you plan to use our aero-open-hand sdk then you might need to add that functionality under aero_hand.py file.

> **Naming tip:** Use `OP_*` for commands initiated by the host and `RX_*` or `OP_RSP_*` for device responses to avoid confusion.

---

## 10) Code Reference (quick lookup)

### Homing API
```c++
bool HOMING_isBusy();
void HOMING_start();
void resetSdToBaseline();
```

### Servo configuration
```c++
struct ServoData {     // one per channel
  uint16_t grasp_count;    // closed endpoint (raw 0..4095)
  uint16_t extend_count;   // open endpoint  (raw 0..4095)
  int8_t   servo_direction; // +1 or -1 direction convention
};
```

### Build‑time hand selection
```c++
// In HandConfig.h: define exactly one
// #define LEFT_HAND
// #define RIGHT_HAND
```

---

## 11) Safety & Best Practices
- Start with **low speeds/torque** when testing new code.
- Ensure the servo supply can deliver peak current without large droops.
- Keep cable lengths short and bus terminations clean; avoid ground loops.
- Avoid sending high‑rate position frames during homing.

---

## 12) Troubleshooting
- **Wrong hand geometry**: Verify your build flags (`-DLEFT_HAND` vs `-DRIGHT_HAND`).
- **Servos drive the wrong way**: Check `servo_direction` for the affected channel or swap extend/grasp counts.
- **Homing stalls**: Inspect current limits/timeouts; make sure mechanics move freely.
- **No serial activity**: Confirm port, baud rate, and that frames are exactly 16 bytes with the correct opcode.

---

## 13) Contribution
We welcome community contributions!

If you would like to improve the Firmware or add new features:

1. Fork and create a feature branch.
2. Add or modify opcodes and handlers as described in Section 9.
3. Include brief unit/bench tests where possible.
4. Commit your changes with clear messages.
5. Push your branch to your fork.
6. Open a PR with a summary and scope of changes.

---

## 14) License
This project is licensed under **Apache License 2.0**

---

## 15) Support 
If you encounter issues or have feature requests:
- Open a [GitHub Issue](https://github.com/TetherIA/aero-open-sdk/issues)
- Contact us at **contact@tetheria.ai**

---

<div align="center">
**Happy building!** Try something new, break things safely, and share what you learn.
If you find this project useful, please give it a star! ⭐

Built with ❤️ by TetherIA.ai
</div>
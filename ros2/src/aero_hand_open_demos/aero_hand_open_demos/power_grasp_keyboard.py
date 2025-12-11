#!/usr/bin/env python3
# Copyright 2025 TetherIA, Inc.
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

import time
from aero_open_sdk.aero_hand import AeroHand

from pynput import keyboard


class KeyboardController:
    def __init__(self):
        self.grasped = False

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.grasped = not self.grasped
                if self.grasped:
                    print("SPACE pressed: moving to GRIP pose")
                else:
                    print("SPACE pressed: moving to ZERO pose")
        except Exception:
            # Don't let any error in the listener kill the program
            pass

def main():
    hand = AeroHand(
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_B8:F8:62:F8:E1:30-if00"
    )

    ZERO_POSE = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    GRIP_POSE = [100.0, 55.0, 30.0, 60.0, 60.0, 60.0, 60.0]

    controller = KeyboardController()

    try:
        while True:
            if controller.grasped:
                hand.set_joint_positions(GRIP_POSE)
            else:
                hand.set_joint_positions(ZERO_POSE)
            time.sleep(0.01)
    except KeyboardInterrupt:
        controller.listener.stop()


if __name__ == "__main__":
    main()
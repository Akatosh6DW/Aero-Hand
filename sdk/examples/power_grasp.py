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

"""
This example demonstrates power grasping with the AeroHand using keyboard input.
Press the SPACE key to toggle between open and close grip poses.
"""

import time
from aero_open_sdk.aero_hand import AeroHand

## This example requires the 'pynput' library.
## You can install it via pip:
##     pip install pynput
try:
    from pynput import keyboard
except ModuleNotFoundError:
    raise SystemExit(
        "\nERROR: 'pynput' is not installed.\n"
        "Install it using:\n"
        "    pip install pynput\n"
    )

class KeyboardController:
    """A simple keyboard controller for the AeroHand.
    Press SPACE to toggle between open and grip poses."""
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
    hand = AeroHand()

    open_pose = [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    grip_pose = [100.0, 55.0, 30.0, 60.0, 60.0, 60.0, 60.0]

    controller = KeyboardController()

    try:
        while True:
            if controller.grasped:
                hand.set_joint_positions(grip_pose)
            else:
                hand.set_joint_positions(open_pose)
            time.sleep(0.01)
    except KeyboardInterrupt:
        controller.listener.stop()


if __name__ == "__main__":
    main()
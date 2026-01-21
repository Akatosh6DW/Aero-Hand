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

from aero_open_sdk.aero_hand import AeroHand


def main():
    hand = AeroHand(
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_D8:3B:DA:45:C7:C0-if00"
    )

    trajectory = [
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),  # ALL OPEN
        ## Pinch fingers one by one
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.5),  # Touch Pinkie
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.25),  # Hold
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.5),  # Touch Ring
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.25),  # Hold
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.5),  # Touch Middle
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.25),  # Hold
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.5),  # Touch Index
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.25),  # Hold
        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),  # Hold
        ## Peace Sign
        ([90.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0], 0.5),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 0.5),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 1.0),
        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),  # Hold
        ## Finger Base Touching
        ([100, 0, 0, 0, 0, 0, 0], 0.5),  # Abd Thumb
        ([100, 0, 0, 0, 0, 0, 0], 0.5),  # Abd Thumb
        ## Touch Ring base
        ([100, 90, 57, 0, 0, 0, 0], 0.5),  # Touch Index
        ## Touch Middle base
        ([34, 90, 52, 0, 0, 0, 0], 0.5),  # Touch Middle
        ## Touch Index base
        ([0, 34, 46, 0, 0, 0, 0], 0.5),  # Touch Index
        ## Touch Middle base
        ([34, 90, 52, 0, 0, 0, 0], 0.5),  # Touch Middle
        ## Touch Ring base
        ([100, 90, 57, 0, 0, 0, 0], 0.5),  # Touch Index
        ## Touch Middle base
        ([34, 90, 52, 0, 0, 0, 0], 0.5),  # Touch Middle
        ## Touch Index base
        ([0, 34, 46, 0, 0, 0, 0], 0.5),  # Touch Index
        ## Open
        ([0, 0, 0, 0, 0, 0, 0], 0.5),  # ALL OPEN
        ([0, 0, 0, 0, 0, 0, 0], 0.5),  # ALL OPEN
        ## THUMB DEXTERITY
        ([0, 0, 90, 0, 0, 0, 0], 0.5),  # Flex MCPs
        ([0, 0, 90, 0, 0, 0, 0], 0.5),  # Flex MCPs
        ([100, 0, 90, 0, 0, 0, 0], 0.5),  # Flex MCPs
        ([100, 0, 90, 0, 0, 0, 0], 0.5),  # Flex MCPs
        ([100, 0, 0, 0, 0, 0, 0], 0.5),  # Flex MCPs
        ([100, 0, 0, 0, 0, 0, 0], 0.5),  # Flex MCPs
        ([0, 0, 0, 0, 0, 0, 0], 0.5),  # ALL OPEN
        ([0, 0, 0, 0, 0, 0, 0], 0.5),  # ALL OPEN
        ## Rockstar Sign
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 0.5),  # Close Middle and Ring Fingers
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 1.0),  # Hold
        ## FINGER SIDE TOUCHING
        ## Ring
        ([92, 63, 23, 0, 0, 54, 0], 0.5),
        ([92, 63, 23, 0, 0, 54, 0], 0.5),
        # ([75, 86, 46, 0, 0, 54, 0], 0.5),
        # ([75, 86, 46, 0, 0, 54, 0], 0.5),
        # Middle
        ([75, 57, 11, 0, 49, 0, 0], 0.5),
        ([75, 57, 11, 0, 49, 0, 0], 0.5),
        ([38, 90, 40, 0, 46, 0, 0], 0.5),
        ([38, 90, 40, 0, 46, 0, 0], 0.5),
        # ([0, 90, 52, 0, 46, 0, 0], 0.5),
        # ([0, 90, 52, 0, 46, 0, 0], 0.5),
        ## Index
        ([63, 57, 11, 46, 0, 0, 0], 0.5),
        ([63, 57, 11, 46, 0, 0, 0], 0.5),
        ([34, 57, 34, 46, 0, 0, 0], 0.5),
        ([34, 57, 34, 46, 0, 0, 0], 0.5),
        ([0, 57, 34, 46, 0, 0, 0], 0.5),
        ([0, 57, 34, 46, 0, 0, 0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),  # ALL OPEN
    ]

    hand.run_trajectory(trajectory)


if __name__ == "__main__":
    main()

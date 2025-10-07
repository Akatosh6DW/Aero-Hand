#!/usr/bin/env python3
import time
from aero_hand_open.aero_hand import AeroHand

if __name__ == "__main__":
    hand = AeroHand(
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_D8:3B:DA:45:CA:08-if00"
    )

    i = 0
    dir = 1
    while True:
        ## Stagnant thumb + Move Fingers
        finger_joint_ranges = [
            ul - ll for ul, ll in zip(hand.joint_upper_limits, hand.joint_lower_limits)
        ]
        joint_pos = [0.0] * 4 + [
            finger_joint_ranges[4 + j] * i / 100.0 for j in range(12)
        ]
        hand.set_joint_positions(joint_pos)
        if dir == 1:
            i += 1
            if i >= 100:
                dir = -1
        else:
            i -= 1
            if i <= 0:
                dir = 1
        time.sleep(0.01)

#!/usr/bin/env python3
from aero_hand_open.aero_hand import AeroHand

if __name__ == "__main__":
    hand = AeroHand(
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_D8:3B:DA:45:CA:08-if00" 
    )

    ## Perform homing
    ## NOTE: While performing homing, robot will not respond to any other commands.
    ## Make sure the hand is in a safe position to perform homing.
    hand.send_homing()
#!/usr/bin/env python3
from aero_open_sdk.aero_hand import AeroHand

if __name__ == "__main__":
    hand = AeroHand(
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_D8:3B:DA:45:C8:1C-if00"
    )

    ## Create a trajectory for the hand to follow
    trajectory = [
        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0),

        ## Pinch fingers one by one
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.5), # Touch Pinkie
        ([100.0, 35.0, 23.0, 0.0, 0.0, 0.0, 50.0], 0.25), # Hold
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.5), # Touch Ring
        ([100.0, 42.0, 23.0, 0.0, 0.0, 52.0, 0.0], 0.25), # Hold
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.5), # Touch Middle
        ([83.0, 42.0, 23.0, 0.0, 50.0, 0.0, 0.0], 0.25), # Hold
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.5), # Touch Index
        ([75.0, 25.0, 30.0, 50.0, 0.0, 0.0, 0.0], 0.25), # Hold

        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5), # Hold

        ## Peace Sign
        ([90.0, 0.0, 0.0, 0.0, 0.0, 90.0, 90.0], 0.5),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 0.5),
        ([90.0, 45.0, 60.0, 0.0, 0.0, 90.0, 90.0], 1.0),

        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5), # Hold

        ## Rockstar Sign
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 0.5), # Close Middle and Ring Fingers
        ([0.0, 0.0, 0.0, 0.0, 90.0, 90.0, 0.0], 1.0), # Hold

        ## Open Palm
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5),
    ]

    hand.run_trajectory(trajectory)
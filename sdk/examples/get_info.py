#!/usr/bin/env python3
import time
from aero_hand_open.aero_hand import AeroHand

if __name__ == "__main__":
    hand = AeroHand(
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_D8:3B:DA:45:CA:08-if00"
    )

    while True:
        motor_current = hand.get_motor_currents()
        print("Motor Currents:", motor_current)
        motor_positions = hand.get_motor_positions()
        print("Motor Positions:", motor_positions)
        motor_speed = hand.get_motor_speed()
        print("Motor Speeds:", motor_speed)
        motor_temperatures = hand.get_motor_temperatures()
        print("Motor Temperatures:", motor_temperatures)
        time.sleep(0.1)

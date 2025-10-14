#!/usr/bin/env python3
import time
from aero_open_sdk.aero_hand import AeroHand

if __name__ == "__main__":
    hand = AeroHand(
        "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_D8:3B:DA:45:C8:1C-if00"
    )

    while True:
        actuator_current = hand.get_actuator_currents()
        print("actuator Currents:", actuator_current)
        actuator_positions = hand.get_actuations()
        print("actuator Positions:", actuator_positions)
        actuator_speeds = hand.get_actuator_speeds()
        print("actuator Speeds:", actuator_speeds)
        actuator_temperatures = hand.get_actuator_temperatures()
        print("actuator Temperatures:", actuator_temperatures)
        time.sleep(0.1)

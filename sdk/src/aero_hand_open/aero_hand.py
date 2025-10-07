#!/usr/bin/env python3
import serial
import struct
import time 

from aero_hand_open.joints_to_actuations import JointsToActuationsModel

## Setup Modes
HOMING_MODE = 0x01
SET_ID_MODE = 0x02
TRIM_MODE = 0x03

## Command Modes
CTRL_POS = 0x11

## Request Modes
GET_ALL = 0x21
GET_POS = 0x22
GET_VEL = 0x23
GET_CURR = 0x24
GET_TEMP = 0x25


## Robot Constants
_JOINT_NAMES = [
    "thumb_cmc_abd",
    "thumb_cmc_flex",
    "thumb_mcp",
    "thumb_ip",
    "index_mcp_flex",
    "index_pip",
    "index_dip",
    "middle_mcp_flex",
    "middle_pip",
    "middle_dip",
    "ring_mcp_flex",
    "ring_pip",
    "ring_dip",
    "pinky_mcp_flex",
    "pinky_pip",
    "pinky_dip",
]

_JOINT_LOWER_LIMITS = [0.0] * 16
_JOINT_UPPER_LIMITS = [100.0, 55.0, 90.0, 90.0] + [90.0] * 12

_ACTUATIONS_LOWER_LIMITS = [0.0, 0.0, -27.7778, 0.0, 0.0, 0.0, 0.0]
_ACTUATIONS_UPPER_LIMITS = [
    100.0,
    131.8906,
    274.9275,
    288.1603,
    288.1603,
    288.1603,
    288.1603,
]

_UINT16_MAX = 65535


class AeroHand:
    def __init__(self, port=None, baudrate=921600):
        ## Connect to serial port
        if port is None:
            ## Lazy initialization for testing without hardware
            self.ser = None
        else:
            self.ser = serial.Serial(port, baudrate, timeout=0.01, write_timeout=0.01)

        self.joint_names = _JOINT_NAMES
        self.joint_lower_limits = _JOINT_LOWER_LIMITS
        self.joint_upper_limits = _JOINT_UPPER_LIMITS

        self.joints_to_actuations_model = JointsToActuationsModel()

    def set_joint_positions(self, positions: list):
        """
        Set the joint positions of the Aero Hand.

        Args:
            positions (list): A list of 16 joint positions. (degrees)
        """
        assert len(positions) == 16, "Expected 16 Joint Positions"

        ## Clamp the positions to the joint limits.
        positions = [
            max(
                self.joint_lower_limits[i],
                min(positions[i], self.joint_upper_limits[i]),
            )
            for i in range(16)
        ]

        ## Convert to actuations
        actuations = self.joints_to_actuations_model.hand_actuations(positions)

        ## Normalize actuation to uint16 range. (0-65535)
        actuations = [
            (actuations[i] - _ACTUATIONS_LOWER_LIMITS[i])
            / (_ACTUATIONS_UPPER_LIMITS[i] - _ACTUATIONS_LOWER_LIMITS[i])
            * _UINT16_MAX
            for i in range(7)
        ]

        self._send_data(CTRL_POS, [int(a) for a in actuations])

    def _wait_for_ack(self, opcode: int, timeout_s: float) -> bytes:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.ser.read(16)
            if len(frame) != 16:
                continue 
            if frame[0] == (opcode & 0xFF) and frame[1] == 0x00:
                return frame[2:]
        raise TimeoutError(f"ACK (opcode 0x{opcode:02X}) not received within {timeout_s}s")
    
    def set_id(self, id: int, current_limit: int):
        """This fn is used by the GUI to set Motor IDs and current limits for the first time."""
        if not (0 <= id <= 253):
            raise ValueError("new_id must be 0..253")
        if not (0 <= current_limit <= 1023):
            raise ValueError("current_limit must be in between 0..1023")
        
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

        payload = [0] * 7
        payload[0] = id & 0xFF   # stored in low byte of word0
        payload[1] = current_limit & 0x03FF
        self._send_data(SET_ID_MODE, payload)
        payload = self._wait_for_ack(SET_ID_MODE, 1.0)
        old_id, new_id, cur_limit = struct.unpack_from("<HHH", payload, 0)
        return {"Old_id": old_id, "New_id": new_id, "Current_limit": cur_limit}
    
    def trim_servo(self, channel: int, degrees: int):
        """This fn is used by the GUI to fine tune the motor positions."""
        if not (0 <= channel <= 14):
            raise ValueError("channel must be 0..14")
        if not (-360 <= degrees <= 360):
            raise ValueError("degrees out of range")
        
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

        payload = [0] * 7
        payload[0] = channel & 0xFFFF
        payload[1] = degrees & 0xFFFF  
        self._send_data(TRIM_MODE, payload)
        payload = self._wait_for_ack(TRIM_MODE, 1.0)
        id, extend = struct.unpack_from("<HH", payload, 0)
        return {"Servo ID": id, "Extend Count": extend}

    def _send_data(self, header: int, payload: list[int] = [0] * 7):
        assert self.ser is not None, "Serial port is not initialized"
        assert len(payload) == 7, "Payload must be a list of 7 integers in Range 0-65535"
        assert all(0 <= v <= 65535 for v in payload), "Payload values must be in Range 0-65535"
        msg = struct.pack("<2B7H", header & 0xFF, 0x00, *(v & 0xFFFF for v in payload))
        self.ser.write(msg)
        self.ser.flush()

    def send_homing(self, timeout_s: float = 175.0):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self._send_data(HOMING_MODE) 
        payload = self._wait_for_ack(HOMING_MODE, timeout_s)
        if all(b == 0 for b in payload):
            return True
        else:
            raise ValueError(f"Unexpected HOMING payload: {payload.hex()}")

    def get_forward_kinematics(self):
        raise NotImplementedError("This method is not yet implemented")

    def get_joint_positions(self):
        raise NotImplementedError("This method is not yet implemented")

    def get_motor_positions(self):
        """
        Get the motor positions from the hand.
        Returns:
            list: A list of 7 motor positions. (degrees)
        """
        self._send_data(GET_POS)
        ## Read the response
        resp = self.ser.read(2 + 7 * 2)  # 2
        data = struct.unpack("<2B7H", resp)
        if data[0] != GET_POS:
            raise ValueError("Invalid response from hand")
        return data[2:]

    def get_motor_currents(self):
        """
        Get the motor currents from the hand.
        Returns:
            list: A list of 7 motor currents. (mA)
        """
        self._send_data(GET_CURR)
        ## Read the response
        resp = self.ser.read(2 + 7 * 2)  # 2
        data = struct.unpack("<2B7h", resp)
        if data[0] != GET_CURR:
            raise ValueError("Invalid response from hand")
        return data[2:]

    def get_motor_temperatures(self):
        """
        Get the motor temperatures from the hand.
        Returns:
            list: A list of 7 motor temperatures. (Degree Celsius)
        """
        self._send_data(GET_TEMP)
        ## Read the response
        resp = self.ser.read(2 + 7 * 2)  # 2
        data = struct.unpack("<2B7H", resp)
        if data[0] != GET_TEMP:
            raise ValueError("Invalid response from hand")
        return data[2:]

    def get_motor_speed(self):
        """
        Get the motor speeds from the hand.
        Returns:
            list: A list of 7 motor speeds. (RPM)
        """
        self._send_data(GET_VEL)
        ## Read the response
        resp = self.ser.read(2 + 7 * 2)  # 2
        data = struct.unpack("<2B7h", resp)
        if data[0] != GET_VEL:
            raise ValueError("Invalid response from hand")
        return data[2:]

    def close(self):
        self.ser.close()

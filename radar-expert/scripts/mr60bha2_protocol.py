#!/usr/bin/env python3
"""
MR60BHA2 Protocol Implementation for FMCW Waveform Control

This script implements the communication protocol for the MR60BHA2 60GHz radar module,
specifically focusing on FMCW (Frequency-Modulated Continuous Wave) waveform control
and configuration.
"""

import serial
import struct
import time
import numpy as np


class MR60BHA2Protocol:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        """
        Initialize communication with MR60BHA2 module
        
        Args:
            port: Serial port connected to the module
            baudrate: Baud rate for communication
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        
    def connect(self):
        """Establish serial connection to the MR60BHA2 module"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"Connected to MR60BHA2 on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to MR60BHA2: {e}")
            return False
    
    def disconnect(self):
        """Close the serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from MR60BHA2")
    
    def configure_fmcw_waveform(self, center_freq=60.0, sweep_freq=1.0, sweep_time=10.0):
        """
        Configure FMCW waveform parameters
        
        Args:
            center_freq: Center frequency in GHz (typically 60 GHz)
            sweep_freq: Sweep bandwidth in GHz
            sweep_time: Sweep time in ms
        """
        if not self.serial_conn:
            print("Not connected to MR60BHA2")
            return False
            
        # Build configuration command
        cmd = self._build_fmcw_config_cmd(center_freq, sweep_freq, sweep_time)
        
        # Send command
        self.serial_conn.write(cmd)
        print(f"FMCW waveform configured: {center_freq}GHz center, {sweep_freq}GHz sweep, {sweep_time}ms sweep time")
        
        # Wait for response
        response = self.serial_conn.read(100)
        if response:
            print(f"Module response: {response.hex()}")
        
        return True
    
    def _build_fmcw_config_cmd(self, center_freq, sweep_freq, sweep_time):
        """
        Build FMCW configuration command packet
        
        Args:
            center_freq: Center frequency in GHz
            sweep_freq: Sweep bandwidth in GHz
            sweep_time: Sweep time in ms
            
        Returns:
            bytes: Configuration command packet
        """
        # Command structure (example - actual protocol may differ):
        # Header (2 bytes) + Center Freq (4 bytes) + Sweep Freq (4 bytes) + Sweep Time (4 bytes) + CRC (2 bytes)
        cmd_bytes = bytearray()
        cmd_bytes.extend([0xAA, 0x55])  # Example header
        cmd_bytes.extend(struct.pack('<f', center_freq))  # Little-endian float
        cmd_bytes.extend(struct.pack('<f', sweep_freq))   # Little-endian float
        cmd_bytes.extend(struct.pack('<f', sweep_time))   # Little-endian float
        
        # Calculate and append CRC (simplified)
        crc = self._calculate_crc(cmd_bytes[2:])  # CRC of payload only
        cmd_bytes.extend(struct.pack('<H', crc))  # Little-endian uint16
        
        return bytes(cmd_bytes)
    
    def _calculate_crc(self, data):
        """
        Calculate CRC for data integrity
        
        Args:
            data: Bytes to calculate CRC for
            
        Returns:
            int: CRC value
        """
        # Simple CRC implementation (actual implementation may vary)
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc
    
    def read_raw_data(self, num_bytes=1024):
        """
        Read raw data from the MR60BHA2 module
        
        Args:
            num_bytes: Number of bytes to read
            
        Returns:
            bytes: Raw data from the module
        """
        if not self.serial_conn:
            print("Not connected to MR60BHA2")
            return b""
        
        return self.serial_conn.read(num_bytes)
    
    def set_detection_sensitivity(self, sensitivity_level=5):
        """
        Set detection sensitivity level
        
        Args:
            sensitivity_level: Sensitivity level (0-10)
        """
        if not self.serial_conn:
            print("Not connected to MR60BHA2")
            return False
            
        # Example command to set sensitivity
        cmd = bytearray([0xBB, 0xCC, sensitivity_level & 0xFF])
        self.serial_conn.write(cmd)
        print(f"Detection sensitivity set to level {sensitivity_level}")
        
        return True


def main():
    """Example usage of the MR60BHA2 protocol implementation"""
    # Initialize the protocol handler
    mr60bha2 = MR60BHA2Protocol('/dev/ttyUSB0')
    
    # Connect to the module
    if not mr60bha2.connect():
        print("Failed to connect to MR60BHA2 module")
        return
    
    try:
        # Configure FMCW waveform
        mr60bha2.configure_fmcw_waveform(
            center_freq=60.0,  # 60 GHz
            sweep_freq=1.0,    # 1 GHz sweep
            sweep_time=10.0    # 10 ms sweep time
        )
        
        # Set detection sensitivity
        mr60bha2.set_detection_sensitivity(7)
        
        # Read some raw data
        print("Reading raw data...")
        raw_data = mr60bha2.read_raw_data(256)
        print(f"Received {len(raw_data)} bytes of raw data")
        
        # Print first 32 bytes as hex
        if len(raw_data) > 0:
            print(f"First 32 bytes: {raw_data[:32].hex()}")
    
    finally:
        # Always disconnect
        mr60bha2.disconnect()


if __name__ == "__main__":
    main()
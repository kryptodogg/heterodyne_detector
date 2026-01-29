#!/usr/bin/env python3
"""
HLK-LD2410 Configuration for 24GHz Radar Module

This script implements the configuration protocol for the HLK-LD2410 24GHz radar module,
focusing on presence and motion detection parameters.
"""

import serial
import struct
import time
import numpy as np


class HLKLD2410Config:
    def __init__(self, port='/dev/ttyUSB1', baudrate=256000):
        """
        Initialize communication with HLK-LD2410 module
        
        Args:
            port: Serial port connected to the module
            baudrate: Baud rate for communication (typically 256000)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.presence_detection_params = {}
        
    def connect(self):
        """Establish serial connection to the HLK-LD2410 module"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"Connected to HLK-LD2410 on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to HLK-LD2410: {e}")
            return False
    
    def disconnect(self):
        """Close the serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Disconnected from HLK-LD2410")
    
    def configure_presence_sensing(self, motion_threshold=0.1, static_threshold=0.05, gate_distance=2.0):
        """
        Configure presence sensing parameters for HLK-LD2410
        
        Args:
            motion_threshold: Motion detection threshold (0.0-1.0)
            static_threshold: Static detection threshold (0.0-1.0)
            gate_distance: Detection distance in meters
        """
        if not self.serial_conn:
            print("Not connected to HLK-LD2410")
            return False
            
        # Store parameters
        self.presence_detection_params['motion_threshold'] = motion_threshold
        self.presence_detection_params['static_threshold'] = static_threshold
        self.presence_detection_params['gate_distance'] = gate_distance
        
        # Build configuration command
        cmd = self._build_presence_config_cmd(motion_threshold, static_threshold, gate_distance)
        
        # Send command
        self.serial_conn.write(cmd)
        print(f"Presence sensing configured: Motion thresh={motion_threshold}, Static thresh={static_threshold}, Distance={gate_distance}m")
        
        # Wait for acknowledgment
        response = self.serial_conn.read(100)
        if response:
            print(f"Module response: {response.hex()}")
        
        return True
    
    def _build_presence_config_cmd(self, motion_threshold, static_threshold, gate_distance):
        """
        Build presence detection configuration command packet
        
        Args:
            motion_threshold: Motion detection threshold (0.0-1.0)
            static_threshold: Static detection threshold (0.0-1.0)
            gate_distance: Detection distance in meters
            
        Returns:
            bytes: Configuration command packet
        """
        # Command structure (example - actual protocol may differ):
        # Header (4 bytes) + Motion Thresh (4 bytes) + Static Thresh (4 bytes) + Gate Dist (4 bytes) + CRC (2 bytes)
        cmd_bytes = bytearray()
        cmd_bytes.extend([0xFD, 0xFC, 0xFB, 0xFA])  # Example header
        cmd_bytes.extend(struct.pack('<f', motion_threshold))  # Little-endian float
        cmd_bytes.extend(struct.pack('<f', static_threshold))  # Little-endian float
        cmd_bytes.extend(struct.pack('<f', gate_distance))     # Little-endian float
        
        # Calculate and append CRC (simplified)
        crc = self._calculate_crc(cmd_bytes[4:])  # CRC of payload only
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
    
    def read_detection_data(self, timeout=5.0):
        """
        Read detection data from the HLK-LD2410 module
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            dict: Detection data including motion, static, and distance information
        """
        if not self.serial_conn:
            print("Not connected to HLK-LD2410")
            return {}
        
        start_time = time.time()
        data = b""
        
        # Read data until we get a complete frame or timeout
        while time.time() - start_time < timeout:
            chunk = self.serial_conn.read(1)
            if chunk:
                data += chunk
                
                # Look for frame header (example: 0xFD 0xFC 0xFB 0xFA)
                if len(data) >= 4 and data[-4:] == b'\xFD\xFC\xFB\xFA':
                    # This indicates a new frame start, so reset
                    data = data[-4:]
                elif len(data) >= 16:  # Minimum frame size
                    # Parse the frame
                    detection_info = self._parse_detection_frame(data)
                    if detection_info:
                        return detection_info
            
            time.sleep(0.01)  # Small delay to prevent busy-waiting
        
        return {}
    
    def _parse_detection_frame(self, frame_data):
        """
        Parse a detection frame from the HLK-LD2410
        
        Args:
            frame_data: Raw frame data
            
        Returns:
            dict: Parsed detection information
        """
        if len(frame_data) < 16:
            return None
            
        try:
            # Example parsing (actual format may differ)
            # Frame: Header + Motion Status (1 byte) + Static Status (1 byte) + Distance (4 bytes) + Energy (4 bytes) + CRC
            motion_status = frame_data[4] if len(frame_data) > 4 else 0
            static_status = frame_data[5] if len(frame_data) > 5 else 0
            distance = struct.unpack('<f', frame_data[6:10])[0] if len(frame_data) >= 10 else 0.0
            energy = struct.unpack('<f', frame_data[10:14])[0] if len(frame_data) >= 14 else 0.0
            
            return {
                'motion_detected': bool(motion_status),
                'static_detected': bool(static_status),
                'distance': distance,
                'energy': energy,
                'timestamp': time.time()
            }
        except (struct.error, IndexError):
            return None
    
    def set_gate_config(self, start_gate=0.5, end_gate=6.0):
        """
        Configure detection gates (distance ranges)
        
        Args:
            start_gate: Start distance for detection (meters)
            end_gate: End distance for detection (meters)
        """
        if not self.serial_conn:
            print("Not connected to HLK-LD2410")
            return False
            
        # Build gate configuration command
        cmd = self._build_gate_config_cmd(start_gate, end_gate)
        
        # Send command
        self.serial_conn.write(cmd)
        print(f"Gate configuration set: {start_gate}m to {end_gate}m")
        
        # Wait for acknowledgment
        response = self.serial_conn.read(100)
        if response:
            print(f"Module response: {response.hex()}")
        
        return True
    
    def _build_gate_config_cmd(self, start_gate, end_gate):
        """
        Build gate configuration command
        
        Args:
            start_gate: Start distance for detection (meters)
            end_gate: End distance for detection (meters)
            
        Returns:
            bytes: Gate configuration command
        """
        cmd_bytes = bytearray()
        cmd_bytes.extend([0xFE, 0xFD, 0xFC, 0xFB])  # Gate config header
        cmd_bytes.extend(struct.pack('<f', start_gate))  # Start gate
        cmd_bytes.extend(struct.pack('<f', end_gate))    # End gate
        
        # Calculate CRC
        crc = self._calculate_crc(cmd_bytes[4:])
        cmd_bytes.extend(struct.pack('<H', crc))
        
        return bytes(cmd_bytes)


def main():
    """Example usage of the HLK-LD2410 configuration"""
    # Initialize the configuration handler
    hlk_ld2410 = HLKLD2410Config('/dev/ttyUSB1')
    
    # Connect to the module
    if not hlk_ld2410.connect():
        print("Failed to connect to HLK-LD2410 module")
        return
    
    try:
        # Configure presence sensing
        hlk_ld2410.configure_presence_sensing(
            motion_threshold=0.15,
            static_threshold=0.08,
            gate_distance=3.0
        )
        
        # Set gate configuration
        hlk_ld2410.set_gate_config(start_gate=0.5, end_gate=5.0)
        
        # Continuously read detection data
        print("Reading detection data (Ctrl+C to stop)...")
        while True:
            detection_data = hlk_ld2410.read_detection_data(timeout=1.0)
            if detection_data:
                print(f"Detection: Motion={detection_data['motion_detected']}, "
                      f"Static={detection_data['static_detected']}, "
                      f"Distance={detection_data['distance']:.2f}m, "
                      f"Energy={detection_data['energy']:.2f}")
            else:
                print("No detection data received")
            
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Always disconnect
        hlk_ld2410.disconnect()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Bluetooth Handler for Radar Modules using Ubuntu's Native Stack

This script demonstrates how to use Ubuntu's native Bluetooth stack
instead of PyBluez for communicating with radar modules like HC-05 and ESP32.
"""

import subprocess
import time
import os
import sys
import threading
from typing import Optional


class UbuntuBluetoothHandler:
    def __init__(self, device_address: str, rfcomm_channel: int = 1):
        """
        Initialize the Bluetooth handler using Ubuntu's native stack
        
        Args:
            device_address: MAC address of the Bluetooth device
            rfcomm_channel: RFCOMM channel to use (default 1)
        """
        self.device_address = device_address
        self.rfcomm_channel = rfcomm_channel
        self.rfcomm_device = f"/dev/rfcomm{rfcomm_channel}"
        self.is_connected = False
        self.connection_thread = None
        
    def check_system_dependencies(self) -> bool:
        """
        Check if required system dependencies are installed
        
        Returns:
            bool: True if all dependencies are available
        """
        required_commands = ['bluetoothctl', 'rfcomm', 'hciconfig']
        
        for cmd in required_commands:
            try:
                subprocess.run(['which', cmd], check=True, stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print(f"Error: Required command '{cmd}' not found.")
                print("Install with: sudo apt-get install bluetooth bluez-tools")
                return False
        
        return True
    
    def check_permissions(self) -> bool:
        """
        Check if the user has necessary permissions to access Bluetooth devices
        
        Returns:
            bool: True if permissions are adequate
        """
        try:
            # Check if user is in dialout group
            groups = subprocess.check_output(['groups'], universal_newlines=True)
            if 'dialout' not in groups:
                print("Warning: User is not in 'dialout' group.")
                print("Add with: sudo usermod -a -G dialout $USER")
                print("Then log out and log back in.")
                return False
            return True
        except subprocess.CalledProcessError:
            print("Could not check user groups")
            return False
    
    def scan_for_devices(self, timeout: int = 10) -> list:
        """
        Scan for nearby Bluetooth devices
        
        Args:
            timeout: Scan duration in seconds
            
        Returns:
            list: Found devices with MAC addresses and names
        """
        print(f"Scanning for Bluetooth devices (timeout: {timeout}s)...")
        
        try:
            # Start scanning
            subprocess.run(['bluetoothctl'], input='scan on\n', text=True, timeout=2)
            time.sleep(timeout)
            
            # Get discovered devices
            result = subprocess.run(['bluetoothctl'], input='devices\n', 
                                  text=True, capture_output=True, timeout=5)
            
            devices = []
            for line in result.stdout.split('\n'):
                if 'Device' in line:
                    parts = line.strip().split(' ', 2)
                    if len(parts) >= 3:
                        mac = parts[1]
                        name = parts[2]
                        devices.append({'mac': mac, 'name': name})
            
            # Stop scanning
            subprocess.run(['bluetoothctl'], input='scan off\n', text=True, timeout=2)
            
            return devices
        except subprocess.TimeoutExpired:
            print("Bluetooth scan timed out")
            return []
        except Exception as e:
            print(f"Error during device scan: {e}")
            return []
    
    def pair_device(self) -> bool:
        """
        Pair with the Bluetooth device using bluetoothctl
        
        Returns:
            bool: True if pairing was successful
        """
        print(f"Attempting to pair with {self.device_address}...")
        
        try:
            # Prepare bluetoothctl commands
            commands = f"""
            scan on
            pair {self.device_address}
            trust {self.device_address}
            quit
            """
            
            # Execute pairing sequence
            result = subprocess.run(
                ['bluetoothctl'], 
                input=commands, 
                text=True, 
                capture_output=True, 
                timeout=30
            )
            
            if 'Pairing successful' in result.stdout:
                print("Device paired successfully")
                return True
            else:
                print(f"Pairing failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Pairing timed out")
            return False
        except Exception as e:
            print(f"Error during pairing: {e}")
            return False
    
    def connect(self) -> bool:
        """
        Establish RFCOMM connection to the device
        
        Returns:
            bool: True if connection was successful
        """
        if not self.check_system_dependencies():
            return False
        
        if not self.check_permissions():
            print("Insufficient permissions. Continuing anyway, but may fail...")
        
        # First, try to pair if not already paired
        if not self.is_device_paired():
            if not self.pair_device():
                print("Failed to pair with device")
                return False
        
        # Bind RFCOMM device
        try:
            print(f"Binding RFCOMM device {self.rfcomm_device} to {self.device_address}...")
            subprocess.run(['sudo', 'rfcomm', 'bind', str(self.rfcomm_channel), 
                          self.device_address, str(self.rfcomm_channel)], check=True)
            print(f"RFCOMM device {self.rfcomm_device} bound successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to bind RFCOMM device: {e}")
            return False
        
        # Wait a moment for the device to appear
        time.sleep(1)
        
        # Verify the device exists
        if os.path.exists(self.rfcomm_device):
            print(f"Connected to {self.device_address} via {self.rfcomm_device}")
            self.is_connected = True
            return True
        else:
            print(f"RFCOMM device {self.rfcomm_device} not found after binding")
            return False
    
    def is_device_paired(self) -> bool:
        """
        Check if the device is already paired
        
        Returns:
            bool: True if device is paired
        """
        try:
            result = subprocess.run(['bluetoothctl'], input='paired-devices\n', 
                                  text=True, capture_output=True, timeout=5)
            
            return self.device_address in result.stdout
        except Exception as e:
            print(f"Error checking pairing status: {e}")
            return False
    
    def send_data(self, data: bytes) -> bool:
        """
        Send data to the connected Bluetooth device
        
        Args:
            data: Bytes to send
            
        Returns:
            bool: True if data was sent successfully
        """
        if not self.is_connected:
            print("Not connected to device")
            return False
        
        try:
            with open(self.rfcomm_device, 'wb') as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            return False
    
    def receive_data(self, num_bytes: int = 1024) -> Optional[bytes]:
        """
        Receive data from the connected Bluetooth device
        
        Args:
            num_bytes: Number of bytes to read
            
        Returns:
            bytes: Received data or None if error occurred
        """
        if not self.is_connected:
            print("Not connected to device")
            return None
        
        try:
            with open(self.rfcomm_device, 'rb') as f:
                data = f.read(num_bytes)
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    
    def disconnect(self):
        """Disconnect and clean up RFCOMM connection"""
        if self.is_connected:
            try:
                subprocess.run(['sudo', 'rfcomm', 'release', str(self.rfcomm_channel)])
                print(f"Released RFCOMM device {self.rfcomm_device}")
                self.is_connected = False
            except Exception as e:
                print(f"Error releasing RFCOMM device: {e}")
    
    def start_continuous_receive(self, callback_func):
        """
        Start continuous receiving in a separate thread
        
        Args:
            callback_func: Function to call with received data
        """
        def receive_loop():
            while self.is_connected:
                data = self.receive_data(1024)
                if data:
                    callback_func(data)
                time.sleep(0.01)  # Small delay to prevent busy-waiting
        
        self.connection_thread = threading.Thread(target=receive_loop, daemon=True)
        self.connection_thread.start()


def main():
    """Example usage of the Ubuntu Bluetooth handler"""
    # Example device address - replace with actual device MAC
    DEVICE_MAC = "AA:BB:CC:DD:EE:FF"  # Replace with actual device
    
    # Create handler instance
    bt_handler = UbuntuBluetoothHandler(DEVICE_MAC)
    
    try:
        # Check system dependencies
        if not bt_handler.check_system_dependencies():
            print("System dependencies not met. Exiting.")
            return
        
        # Attempt connection
        if not bt_handler.connect():
            print("Failed to connect to device. Exiting.")
            return
        
        # Send some test data
        test_data = b"Hello Bluetooth Radar Module!"
        if bt_handler.send_data(test_data):
            print(f"Sent: {test_data.decode('utf-8', errors='ignore')}")
        
        # Receive some data
        received = bt_handler.receive_data(1024)
        if received:
            print(f"Received: {received.decode('utf-8', errors='ignore')}")
        
        # Example of continuous receiving
        def data_callback(data):
            print(f"Callback received: {data.decode('utf-8', errors='ignore')}")
        
        print("Starting continuous receive (Ctrl+C to stop)...")
        bt_handler.start_continuous_receive(data_callback)
        
        # Keep the program running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
    
    finally:
        # Always disconnect
        bt_handler.disconnect()


if __name__ == "__main__":
    main()
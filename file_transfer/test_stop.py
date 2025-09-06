import serial
import time

infrared_port = 'COM6'
# Initialize serial connection
ser = serial.Serial(infrared_port, 115200, timeout=1)  # Adjust '/dev/ttyS0' as needed
time.sleep(1)
ser.write(b'2')

import serial
import time

infrared_port = 'COM6'
# Initialize serial connection
ser = serial.Serial(infrared_port, 115200, timeout=1)  # Adjust '/dev/ttyS0' as needed
time.sleep(1)
while True:
    ser.write(b'1')
    time.sleep(5)
    ser.write(b'2')
    time.sleep(5)

import serial
import time
LED_arduino_port = "COM4"
LED_arduino = serial.Serial(LED_arduino_port, 9600)
time.sleep(3)
print("Running. Press Ctrl+C to interrupt.")
LED_arduino.write(b'0')
time.sleep(1)
LED_arduino.write(b'1')
try:
    while True:
        pass
except KeyboardInterrupt:
    LED_arduino.write(b'0')
    print("Interrupted. Exiting.")

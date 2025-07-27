import boto3
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import datetime as datetime
import pyautogui
from queue import Queue
import serial
import pandas as pd
from scanning_utils import get_size, clean_line, is_file_stable

#       steps per rev      num revs
SCAN_TIME = 3200 * 500 * 2 * 6 * 1e-6
#                 step delay      in sec


# HARD LIMITS ON START AND END DISTANCE
CLOSE_START_DISTANCE = 0.888
FAR_END_DISTANCE = 1.035
# Default start and end distance if none are provided at start, otherwise it uses the previous scan distance
start_distance = 0.888
end_distance = 1.035

current_spider_name = "default"
num_scans = 0

# Initialize S3 client
s3_client = boto3.client('s3')

# Replace with your S3 bucket name
bucket_name = 'spider-videos'

# Set waiting time (seconds) between videos
wait_time = 0

# Choose whether to delete video upon upload
delete_video = True

# Choose port Arduino is connected to
arduino_port = "COM6"

arduino = serial.Serial(arduino_port, 115200)

LED_arduino_port = "COM4"

try:
    LED_arduino = serial.Serial(LED_arduino_port, 9600)
    print("Laser Arduino connected.")
except:
    LED_arduino = None
    print("Laser Arduino not connected.")
time.sleep(0.5)

brightness_dict = {
    0: 0,
    1: 190,
    2: 210,
    3: 230,
    4: 255,
}

infrared_port = 'COM5'
# Initialize serial connection
try:
    ser = serial.Serial(infrared_port, 9600, timeout=1)  # Adjust '/dev/ttyS0' as needed
    print("Infrared Arduino connected.")
except:
    ser = None
    print("Infrared Arduino not connected.")

# Command to initiate measurement
measure_command = bytearray([0x80, 0x06, 0x03, 0x77])


upload_batch_size = 4  # number of scans to batch upload

def send_ready_signal(speed=1):
    if speed == 1:
        arduino.write(b'1')
    elif speed == 0:
        arduino.write(b'4')

def send_back_signal():
    arduino.write(b'2')

def send_stop_signal():
    arduino.write(b'3')


def send_LED_brightness(brightness_level):
    if LED_arduino is None:
        return
    print(f"Brightness {brightness_level}")
    if brightness_level == 0:
        LED_arduino.write(b'0')
    elif brightness_level == 1:
        LED_arduino.write(b'1')
    elif brightness_level == 2:
        LED_arduino.write(b'2')
    elif brightness_level == 3:
        LED_arduino.write(b'3')
    elif brightness_level == 4:
        LED_arduino.write(b'4')
    else:
        raise Exception("Invalid brightness value.")


processed_files = set()

class UploadEventHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        super(UploadEventHandler, self).__init__()
        self.upload_queue = []
        self.brightness_counter = 0
        self.time_info = []
        self.distance_info = []
        self.csv_file_name = f"DEFAULT NAME"

    def on_created(self, event):
        print(f"Current event: {event.src_path}")
        
        if not event.is_directory and event.src_path.lower().endswith(".mp4") and event.src_path not in processed_files:
            current_brightness = brightness_dict[self.brightness_counter + 1]
            self.brightness_counter = (self.brightness_counter + 1) % 4

            # Get timestamp from the file name
            file_name = os.path.basename(event.src_path)
            current_time = file_name.split(".")[0]
            self.csv_file_name = f"{current_spider_name} {current_brightness} distance data {current_time}"

            processed_files.add(event.src_path)
            print(f"Detected new file: {event.src_path}")
            self.upload_queue.append((event.src_path, current_brightness))
            print("Adding to queue...")
            print("Queue state:", self.upload_queue)



def upload_file(file_path, brightness=None, error_queue=None):
    file_name = os.path.basename(file_path)
    while not is_file_stable(file_path, wait_time=5, retries=3):
        pass
    print(f'Uploading {file_name} to S3 bucket {bucket_name}...')
    print(f'File size: {get_size(file_path)}')
    start_time = time.time()
    try:
        ext = os.path.splitext(file_path)[1]
        if "mp4" in ext:
            save_name = current_spider_name + " " + str(brightness) + " " + os.path.basename(file_path)
            print(f"Save name: {save_name}")
        else:  # .csv
            save_name = os.path.basename(file_path)
        s3_client.upload_file(file_path, bucket_name, current_spider_name + "/" + save_name)
        print(f"{file_name} uploaded.")
        end_time = time.time()
        print(f"Upload took {end_time - start_time:.2f} seconds to complete.")
        if delete_video:
            print("Removing file...")
            os.remove(file_path)
            print(f"{file_name} removed.")
        
        
    except Exception as e:
        print(f"Error during upload: {e}")
        print("Adding back to queue...")
        if error_queue is not None:
            error_queue.append((file_path, brightness))


def check_input(user_input):
    """Check if the input is valid or not"""
    global current_spider_name, num_scans, running, start_distance, end_distance, cycle_brightness

    try:
        if len(user_input.split()) == 2:
            current_spider_name, num_scans = tuple(user_input.split())
        elif len(user_input.split()) == 4:
            current_spider_name, num_scans, start_distance, end_distance = tuple(user_input.split())

            start_distance = max(float(start_distance), CLOSE_START_DISTANCE)
            end_distance = min(float(end_distance), FAR_END_DISTANCE)
        else:
            raise ValueError("Input must be either spider_name num_scans or spider_name num_scans start_dist end_dist")

        if num_scans == "inf":
            num_scans = float("inf")
        else:
            num_scans = int(num_scans)    

        cycle_brightness = 4
        running = True
        print(f"Current running settings: {current_spider_name, num_scans, start_distance, end_distance}")

    except Exception as e:
        print(e)
        print("INVALID INPUT")
    return current_spider_name, num_scans, start_distance, end_distance


def read_infrared_distance():
    if ser is None:
        return None
    ser.write(measure_command)
    start_timeout = time.time()
    while ser.in_waiting < 11:
        if time.time() - start_timeout > 1:
            ser.write(measure_command)
            start_timeout = time.time()
    response = ser.read(11)
    # Process the response to extract distance
    if response[0] == 0x80 and response[1] == 0x06:
        distance=(response[3]-0x30)*100+(response[4]-0x30)*10+(response[5]-0x30)*1+(response[7]-0x30)*0.1+(response[8]-0x30)*0.01+(response[9]-0x30)*0.001
    else:
        print("Invalid response")
        return None
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return distance


def reset_position():
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
    if LED_arduino is not None:
        LED_arduino.reset_input_buffer()
        LED_arduino.reset_output_buffer()
    print("reset buffers")
    print("Resetting position...")
    send_LED_brightness(0)
    

    distance = None
    while distance == None:
        distance = read_infrared_distance()
        print(distance)

    send_ready_signal()
    time.sleep(0.05)
    send_back_signal()

    # while True:
    #     line = clean_line(arduino.readline().decode('utf-8').strip())
    #     try:
    #         distance = float(line)
    #         print(distance)
    #         break
    #     except Exception as e:
    #         print("Invalid data received (reset)")
    #         print(e)
    #         continue
    # print(f"Distance received: {distance}")

    while distance >= start_distance:
        distance = read_infrared_distance()
        print(distance)

    print("Sent stop signal.")
    send_stop_signal()
    print("Position reset.")
    time.sleep(1)


if __name__ == "__main__":

    send_LED_brightness(1)

    user_input = input("Enter input with format (current_spider_name num_scans (optional)start_distance (optional)end_distance): ")
    
    current_spider_name, num_scans, start_distance, end_distance = check_input(user_input)

    print("Set up the observer")
    path = os.path.expanduser('~/Videos')

    event_handler = UploadEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)

    print("Start the observer")
    observer.start()

    recording_begin_time = time.time()

    cycle_brightness = 4
    cnt_scan = 0

    if current_spider_name == "read":
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        if LED_arduino is not None:
            LED_arduino.reset_input_buffer()
            LED_arduino.reset_output_buffer()
        distance = None
        while distance == None:
            distance = read_infrared_distance()
            print(distance)
        print(f"Distance received: {distance}")
        while True:
            distance = read_infrared_distance()
            print(distance)
        exit()

    #reset_position()

    try:
        while True:
            
            if cnt_scan == num_scans:
                running = False
            if not running:
                print("Scanning complete.")
                break
            
            print(f"Scan {cnt_scan + 1}")
            print(f"Current runtime: {str(datetime.timedelta(seconds=(time.time() - recording_begin_time)))}")
            
            current_brightness = brightness_dict[cycle_brightness]
            send_LED_brightness(cycle_brightness)

            # distance = None
            # while distance == None:
            #     distance = read_infrared_distance()
            #     print(distance)
            
            print("Sending hotkeys.")
            pyautogui.hotkey('ctrl', 'f11', interval=0.1)
            print(f"Video recording start: scan {cnt_scan + 1}.")

            # arduino.reset_input_buffer()
            time.sleep(0.1)
            send_LED_brightness(cycle_brightness)
            time.sleep(.5)
            print("ready signal sent")
            send_ready_signal(speed=0)
            time.sleep(0.25)
            start_timer = time.perf_counter()
            time.sleep(SCAN_TIME + 1)
            print("back signal sent")
            send_back_signal()
            send_LED_brightness(0)
            if current_spider_name != "reset":
                pyautogui.hotkey('ctrl', 'f12', interval=0.1)
                print(f"Video recording end: scan {cnt_scan + 1}.")
            # while distance >= start_distance:
            #     distance = read_infrared_distance()
            #     print(distance)
            # print("sending stop signal")
            # send_stop_signal()

            time.sleep(SCAN_TIME + 1)
            print("Distance data sample for", event_handler.csv_file_name)
            print(data_df.head())
            print("Total number of logs:", len(data_df))
            data_file = os.path.expanduser(f"~/Documents/distance_data_holder/{event_handler.csv_file_name}.csv")
            data_df.to_csv(data_file)
            event_handler.time_info.clear()
            event_handler.distance_info.clear()
    
            upload_file(data_file, error_queue=event_handler.upload_queue)  # upload csv file

            if current_spider_name != "reset" and num_scans == float("inf") and (cnt_scan % 4 == 3): # last statement creates batched scans
                print(f'Finished scan {cnt_scan + 1}, waiting for {wait_time} seconds...')
                time.sleep(wait_time)


            cnt_scan += 1
            cycle_brightness = (cycle_brightness - 1) if (cycle_brightness - 1) > 0 else 4

            print("Showing current queue state:", event_handler.upload_queue)
            if len(event_handler.upload_queue) == upload_batch_size:
                print("Batch upload detected.")
                upload_start_time = time.time()

                while event_handler.upload_queue:
                    path, current_brightness = event_handler.upload_queue.pop()                    
                    upload_file(path, brightness=current_brightness, error_queue=event_handler.upload_queue)
                print("Batch upload complete.")
                upload_end_time = time.time()
                print(f"Batch upload took {upload_end_time - upload_start_time:.2f} seconds.")
                reset_position()

                assert not event_handler.upload_queue, "Queue not empty after batch upload."

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT; FINALIZING UPLOADS...")
        send_stop_signal()
        pyautogui.hotkey('ctrl', 'f12', interval=0.1)

    finally:
        print("Shutting down processes...")
        print("Batch upload detected.")
        upload_start_time = time.time()

        while event_handler.upload_queue:
            path, current_brightness = event_handler.upload_queue.pop()                    
            upload_file(path, brightness=current_brightness, error_queue=event_handler.upload_queue)
        print("Batch upload complete.")
        upload_end_time = time.time()
        print(f"Batch upload took {upload_end_time - upload_start_time:.2f} seconds.")
        #reset_position()
        observer.stop()
        observer.join()
        print("Processes stopped.")
        send_stop_signal()

    print("Reached end of program")

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


# HARD LIMITS ON START AND END DISTANCE
CLOSE_START_DISTANCE = 1.565
FAR_END_DISTANCE = 1.680

# Default start and end distance if none are provided at start, otherwise it uses the previous scan distance
start_distance = 1.565
end_distance = 1.680

current_spider_name = "default"
num_scans = 0

# Initialize S3 client
s3_client = boto3.client('s3')

# Replace with your S3 bucket name
bucket_name = 'spider-videos'


# Set video length (seconds)
video_length = 15 # redundant if determined by arduino

# Set waiting time (seconds) between videos
wait_time = 0

# Choose whether to delete video upon upload
delete_video = True

# Choose port Arduino is connected to
arduino_port = "COM3"

arduino = serial.Serial(arduino_port, 115200)

LED_arduino_port = "COM4"

LED_arduino = serial.Serial(LED_arduino_port, 9600)
time.sleep(3)

brightness_dict = {
    0: 0,
    1: 190,
    2: 210,
    3: 230,
    4: 255,
}


upload_batch_size = 4  # number of scans to batch upload

def clean_line(line: str):
    i = 0
    for c in line: 
        if not c.isdigit() and not c == '.':
            break
        i += 1
    return line[:i]

# def wait_for_arduino():
#     while True:
#         if arduino.in_waiting > 0:
#             if arduino.read() == b'd':
#                 break


def send_ready_signal():
    arduino.write(b'1')


def send_back_signal():
    arduino.write(b'2')


def send_stop_signal():
    arduino.write(b'3')


def send_LED_brightness(brightness_level):
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


# def wait_arduino_recovery():
#     while True:
#         if arduino.in_waiting > 0:
#             if arduino.read() == b's':
#                 print("Backward finished.")
#                 break


def get_size(path):
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} bytes"
    elif size < pow(1024,2):
        return f"{round(size/1024, 2)} KB"
    elif size < pow(1024,3):
        return f"{round(size/(pow(1024,2)), 2)} MB"
    elif size < pow(1024,4):
        return f"{round(size/(pow(1024,3)), 2)} GB"


processed_files = set()

class UploadEventHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        super(UploadEventHandler, self).__init__()
        self.upload_queue = []
        self.brightness_counter = 0
        self.time_info = []
        self.distance_info = []

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

        cycle_brightness = 0
        running = True
        print(f"Current running settings: {current_spider_name, num_scans, start_distance, end_distance}")

    except Exception as e:
        print(e)
        print("INVALID INPUT")
    return current_spider_name, num_scans, start_distance, end_distance


def is_file_stable(file_path, wait_time=2, retries=3):
    last_size = -1

    while True:
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            return False
        
        if current_size == last_size:
            retries -= 1
        
        last_size = current_size

        if retries <= 0:
            return True
        
        time.sleep(wait_time)


def reset_position():
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
    LED_arduino.reset_input_buffer()
    LED_arduino.reset_output_buffer()
    print("reset buffers")
    print("Resetting position...")
    send_LED_brightness(0)
    print("Sent back signal.")
    send_ready_signal()
    time.sleep(0.05)
    send_back_signal()

    while True:
        line = clean_line(arduino.readline().decode('utf-8').strip())
        try:
            distance = float(line)
            print(distance)
            break
        except Exception as e:
            print("Invalid data received (reset)")
            print(e)
            continue
    print(f"Distance received: {distance}")

    while distance >= start_distance:
        #if arduino.in_waiting > 0:
        while True:
            line = clean_line(arduino.readline().decode('utf-8').strip())
            try:
                distance = float(line)
                print(distance)
                break
            except Exception as e:
                print("Invalid data received")
                print(e)
                continue
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

    cycle_brightness = 0
    cnt_scan = 0

    reset_position()

    try:
        while True:
            
            if cnt_scan == num_scans:
                running = False
            if not running:
                print("Scanning complete.")
                break
            
            print(f"Scan {cnt_scan + 1}")
            print(f"Current runtime: {str(datetime.timedelta(seconds=(time.time() - recording_begin_time)))}")
            
            current_brightness = brightness_dict[cycle_brightness + 1]
            send_LED_brightness(cycle_brightness + 1)
            
            print("Sending hotkeys.")
            pyautogui.hotkey('ctrl', 'f11', interval=0.1)
            print(f"Video recording start: scan {cnt_scan + 1}.")

            arduino.reset_input_buffer()
            time.sleep(.1)
            print("ready signal sent")
            send_ready_signal()
            while True:
                line = clean_line(arduino.readline().decode('utf-8').strip())
                try:
                    distance = float(line)
                    print(distance)
                    break
                except Exception as e:
                    print("Invalid data received (1)")
                    print(e)
                    continue
            print(f"Distance received: {distance}")
            start_timer = time.perf_counter()

            while distance <= end_distance:
                #if arduino.in_waiting > 0:
                while True:
                    line = clean_line(arduino.readline().decode('utf-8').strip())  
                    try:
                        distance = float(line)
                        print(distance)
                        event_handler.time_info.append(time.perf_counter() - start_timer)
                        event_handler.distance_info.append(distance)
                        break
                    except Exception as e:
                        print("Invalid data received (2)")
                        print(e)
                        continue
            print("sending back signal")
            send_back_signal()
            send_LED_brightness(0)
            if current_spider_name != "reset":
                pyautogui.hotkey('ctrl', 'f12', interval=0.1)
                print(f"Video recording end: scan {cnt_scan + 1}.")
            while distance >= start_distance:
                #if arduino.in_waiting > 0:
                while True:
                    line = clean_line(arduino.readline().decode('utf-8').strip())
                    try:
                        distance = abs(float(line))
                        print(distance)
                        break
                    except Exception as e:
                        print("Invalid data received (3)")
                        print(e)
                        continue
            print("sending stop signal")
            send_stop_signal()

            # Make csv file
            data_df = pd.DataFrame({"Time": event_handler.time_info, "Distance": event_handler.distance_info})
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
            cycle_brightness = (cycle_brightness + 1) % 4

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
        reset_position()
        observer.stop()
        observer.join()
        print("Processes stopped.")
        send_stop_signal()

    print("Reached end of program")

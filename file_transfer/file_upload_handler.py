import boto3
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import datetime as datetime
import pyautogui
from queue import Queue
import threading
import serial
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# HARD LIMITS ON START AND END DISTANCE
CLOSE_START_DISTANCE = 1.565
FAR_END_DISTANCE = 1.675

# Default start and end distance if none are provided at start, otherwise it uses the previous scan distance
start_distance = 1.565
end_distance = 1.675

current_spider_name = "default"
num_scans = 0

# Initialize S3 client
s3_client = boto3.client('s3')

# Replace with your S3 bucket name
bucket_name = 'spider-videos'


# Set video length (seconds)
video_length = 15 # redundant if determined by arduino

# Set waiting time (seconds) between videos
wait_time = 300

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


# global records of distance and times
distance_info = []
time_info = []


def wait_for_arduino():
    while True:
        if arduino.in_waiting > 0:
            if arduino.read() == b'd':
                #print("Arduino finished.")
                break


def send_ready_signal():
    arduino.write(b'1')
    #print("Ready signal sent.")


def send_back_signal():
    arduino.write(b'2')
    #print("Back signal sent.")


def send_stop_signal():
    arduino.write(b'3')
    #print("Stop signal sent.")


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


def wait_arduino_recovery():
    while True:
        if arduino.in_waiting > 0:
            if arduino.read() == b's':
                print("Backward finished.")
                break


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
    def __init__(self, upload_queue) -> None:
        super(UploadEventHandler, self).__init__()
        self.upload_queue = upload_queue
        self.brightness_counter = 0

    def on_created(self, event):
        print(f"Current event: {event.src_path}")
        time.sleep(5)
        if not event.is_directory and event.src_path.lower().endswith(".mp4") and event.src_path not in processed_files:
            current_brightness = brightness_dict[self.brightness_counter + 1]
            self.brightness_counter = (self.brightness_counter + 1) % 4

            # Get timestamp from the file name
            file_name = os.path.basename(event.src_path)
            current_time = file_name.split(" ")[-1].split(".")[0]
            csv_file_name = f"distance_data {current_time} {current_spider_name} {current_brightness}"

            # Make csv file
            data_df = pd.DataFrame({"Time": time_info, "Distance": distance_info})
            data_file = os.path.expanduser(f"~/Documents/distance_data_holder/{csv_file_name}.csv")
            data_df.to_csv(data_file)
            time.sleep(5)
            executor.submit(upload_file, data_file)
            
            processed_files.add(event.src_path)
            print(f"Detected new file: {event.src_path}")
            self.upload_queue.put((event.src_path, current_brightness))
            print("Adding to queue...")
            print("Queue state:", self.upload_queue.queue)


def upload_worker(upload_queue):
    while not shutdown_flag.is_set():
        file_path, current_brightness = upload_queue.get()
        print(f"Uploading {file_path} with brightness {current_brightness}")
        if file_path is None:
            break
        try:
            upload_file(file_path, brightness=current_brightness)
        except Exception:
            print("Error uploading.")
        finally:
            upload_queue.task_done()


def upload_file(file_path, brightness=None):
    file_name = os.path.basename(file_path)
    #print("Verifying file completeness...")
    while not is_file_stable(file_path, wait_time=5, retries=3):
        pass
    print(f'Uploading {file_name} to S3 bucket {bucket_name}...')
    print(f'File size: {get_size(file_path)}')
    start_time = time.time()
    try:
        ext = os.path.splitext(file_path)[1]
        if "mp4" in ext:
            save_name = os.path.basename(file_path).replace(".mp4", "") + " " + current_spider_name + " " + str(brightness) + ".mp4"
            print(f"Save name: {save_name}")
        else:  # .csv
            save_name = os.path.basename(file_path)
            # save_name = os.path.basename(file_path).replace(".csv", "") + " " + brightness + ".csv"
        s3_client.upload_file(file_path, bucket_name, save_name)
        print(f"{file_name} uploaded.")
        end_time = time.time()
        print(f"Upload took {end_time - start_time:.2f} seconds to complete.")
        if delete_video:
            print("Removing file...")
            os.remove(file_path)
        
    except Exception as e:
        print(f"Error during upload: {e}")


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
        #print(f"Checking integrity of {file_path}... current size: {current_size}")
        
        if current_size == last_size:
            retries -= 1
            #print(f"File {file_path} stable; checking {retries} more times.")
        
        last_size = current_size

        if retries <= 0:
            return True
        
        time.sleep(wait_time)


# TODO: fix this function
def reset_position():
    print("Resetting position...")
    send_LED_brightness(0)
    # send_ready_signal()
    send_back_signal()

    line = arduino.readline().decode('utf-8').strip()
    line = arduino.readline().decode('utf-8').strip()
    distance = float(line)

    while distance >= start_distance:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8').strip()
            try:
                distance = float(line)
                # print(f"Distance: {distance} meters")
                # time_info.append(time.perf_counter() - start_timer)
                # distance_info.append(distance)
            except Exception as e:
                print("Invalid data received")
                print(e)
                send_stop_signal()
    # wait_for_arduino()
    # send_back_signal()
    # wait_arduino_recovery()
    send_stop_signal()
    print("Position reset.")
    time.sleep(1)


if __name__ == "__main__":
    user_input = input("Enter input with format (current_spider_name num_scans (optional)start_distance (optional)end_distance): ")
    
    current_spider_name, num_scans, start_distance, end_distance = check_input(user_input)

    shutdown_flag = threading.Event()

    print("Set up the observer")
    path = os.path.expanduser('~/Videos')
    upload_queue = Queue()
    event_handler = UploadEventHandler(upload_queue)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)

    executor = ThreadPoolExecutor(max_workers=5)

    print("Start the observer")
    observer.start()

    uploader_thread = threading.Thread(target=upload_worker, args=(upload_queue,))
    uploader_thread.start()

    recording_begin_time = time.time()

    cycle_brightness = 0
    cnt_scan = 0

    # reset_position()

    try:
        while True:
            if cnt_scan == num_scans:
                running = False
            if not running:
                print("Scanning complete.")
                break

            print(f"Scan {cnt_scan + 1}")
            print(f"Current runtime: {str(datetime.timedelta(seconds=(time.time() - recording_begin_time)))}")
            
            # time_info = []
            # distance_info = []

            # if current_spider_name != "reset":
            current_brightness = brightness_dict[cycle_brightness + 1]
            send_LED_brightness(cycle_brightness + 1)
            
            pyautogui.hotkey('ctrl', 'f11', interval=0.1)
            print("Video recording start.")
            
            send_ready_signal()
            print("another")
            start_timer = time.perf_counter()
            # while True:
            #     line = arduino.readline().decode('utf-8').strip()
            #     if line != "":
            #         break
            line = arduino.readline().decode('utf-8').strip()
            print("bb")
            try:
                distance = float(line)
                #print(f"Distance: {distance} meters")
            except Exception as e:
                print("Invalid data received")
                print(e)
                send_stop_signal()

            print("cc")

            # Prepare distance data
            while distance <= end_distance:
                print("a", time.time())
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    try:
                        distance = float(line)
                        # print(f"Distance: {distance} meters")
                        time_info.append(time.perf_counter() - start_timer)
                        distance_info.append(distance)
                    except Exception as e:
                        print("Invalid data received")
                        print(e)
                        send_stop_signal()
                
            send_back_signal()
            send_LED_brightness(0)
            if current_spider_name != "reset":
                pyautogui.hotkey('ctrl', 'f12', interval=0.1)
                print("Video recording end.")
            while distance >= start_distance:
                print("b", time.time(), distance, start_distance)
                if arduino.in_waiting > 0:
                    line = arduino.readline().decode('utf-8').strip()
                    # print(line)
                    try:
                        distance = abs(float(line))
                        # print(f"Distance: {distance} meters")
                    except ValueError:
                        print("Invalid data received")
                        send_stop_signal()
            send_stop_signal()
            print("after stop signal")

            if current_spider_name != "reset" and num_scans == float("inf") and not cycle_brightness: # last statement creates batched scans
                print(f'last cycle\'s brightness mod 4: {cycle_brightness}')
                time.sleep(wait_time)

            cnt_scan += 1

    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT; FINALIZING UPLOADS...")
        pyautogui.hotkey('ctrl', 'f12', interval=0.1)
        shutdown_flag.set()
        upload_queue.put(None)

    finally:
        print("Shutting down processes...")
        uploader_thread.join()
        print("aa")
        observer.stop()
        print("bb")
        observer.join()
        print("Processes stopped.")

    print("Reached end of program")

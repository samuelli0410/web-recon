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


default_spider_name = "default"
current_spider_name = "default"
num_scans = 0



# Initialize S3 client
s3_client = boto3.client('s3')

# Replace with your S3 bucket name
bucket_name = 'spider-videos'


# Set video length (seconds)
video_length = 15 # redundant if determined by arduino

# Set waiting time (seconds) between videos
wait_time = 1

# Choose whether to delete video upon upload
delete_video = True

# Choose port Arduino is connected to
arduino_port = "COM3"

arduino = serial.Serial(arduino_port, 115200)
time.sleep(3)


def wait_for_arduino():
    while True:
        if arduino.in_waiting > 0:
            if arduino.read() == b'd':
                print("Arduino finished.")
                break

def send_ready_signal():
    arduino.write(b'1')
    print("Ready signal sent.")

def send_back_signal():
    arduino.write(b'2')
    print("Back signal sent.")

def send_stop_signal():
    arduino.write(b'3')
    print("Stop signal sent.")

def wait_arduino_recovery():
    while True:
        if arduino.in_waiting > 0:
            if arduino.read() == b's':
                print("Backward finished.")
                break

processed_files = set()

class UploadEventHandler(FileSystemEventHandler):
    def __init__(self, upload_queue) -> None:
        super(UploadEventHandler, self).__init__()
        self.upload_queue = upload_queue

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".mp4") and event.src_path not in processed_files:
            processed_files.add(event.src_path)
            print(f"Detected new file: {event.src_path}")
            self.upload_queue.put(event.src_path)
            print("Adding to queue...")
            print("Queue state:", self.upload_queue.queue)

def upload_worker(upload_queue):
    while not shutdown_flag.is_set():
        file_path = upload_queue.get()
        if file_path is None:
            break
        try:
            upload_file(file_path)
        except Exception:
            print("Error uploading.")
        finally:
            upload_queue.task_done()


def upload_file(file_path):
    file_name = os.path.basename(file_path)
    print("Verifying file completeness...")
    while not is_file_stable(file_path, wait_time=5, retries=3):
        pass
    print(f'Uploading {file_name} to S3 bucket {bucket_name}...')
    start_time = time.time()
    try:
        new_file_name = file_path.replace(".mp4", "") + " " + current_spider_name + ".mp4"
        os.rename(file_path, new_file_name)
        s3_client.upload_file(new_file_name, bucket_name, file_name)
        print(f'{new_file_name} uploaded.')
        end_time = time.time()
        print(f"Upload took {end_time - start_time} seconds to complete.")
        if delete_video:
            print("Removing file...")
            os.remove(new_file_name)
        
    except Exception as e:
        print(f"Error during upload: {e}")

running = False

def input_thread_function():
    global current_spider_name, num_scans

    while True:
        user_input = input("Enter input: ")
        current_spider_name, num_scans= tuple(user_input.split())
        num_scans = int(num_scans)

def is_file_stable(file_path, wait_time=2, retries=3):
    last_size = -1

    while True:
        
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            return False
        
        print(f"Checking integrity of {file_path}... current size: {current_size}")
        
        if current_size == last_size:
            retries -= 1
            print(f"File {file_path} stable; checking {retries} more times.")
        
        last_size = current_size

        if retries <= 0:
            return True
        
        time.sleep(wait_time)


shutdown_flag = threading.Event()

# Set up the observer
path = os.path.expanduser('~/Videos')
upload_queue = Queue()
event_handler = UploadEventHandler(upload_queue)
observer = Observer()
observer.schedule(event_handler, path, recursive=False)

executor = ThreadPoolExecutor(max_workers=5)

input_thread = threading.Thread(target=input_thread_function)
input_thread.daemon = True
input_thread.start()



# Start the observer
observer.start()

uploader_thread = threading.Thread(target=upload_worker, args=(upload_queue,))
uploader_thread.start()


print(f'Monitoring folder {path} for new video files...')
recording_begin_time = time.time()

try:
    while True:
        if num_scans > 0:
            num_scans = num_scans - 1
        else:
            current_spider_name = default_spider_name
            
        print(f"Current runtime: {str(datetime.timedelta(seconds=(time.time() - recording_begin_time)))}")
        
        time_info = []
        distance_info = []

        pyautogui.hotkey('ctrl', 'f11', interval=0.1)
    
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        file_name = f"distance_data {current_time}.csv"
        print("Video recording start.")
        send_ready_signal()
        start_timer = time.perf_counter()
        line = arduino.readline().decode('utf-8').strip()
        
        try:
            distance = float(line)
            #print(f"Distance: {distance} meters")
        except Exception as e:
            print("Invalid data received")
            print(e)

        
        while distance <= 1.675:
            if arduino.in_waiting > 0:
                
                line = arduino.readline().decode('utf-8').strip()
                try:
                    distance = float(line)
                    #print(f"Distance: {distance} meters")
                    time_info.append(time.perf_counter() - start_timer)
                    distance_info.append(distance)
                except Exception as e:
                    print("Invalid data received")
                    print(e)
            
        send_back_signal()
        pyautogui.hotkey('ctrl', 'f12', interval=0.1)
        print("Video recording end.")
        data_df = pd.DataFrame({"Time": time_info, "Distance": distance_info})
        print(data_df)
        data_file = os.path.expanduser(f"~/Documents/distance_data_holder/{file_name}")
        data_df.to_csv(data_file)
        while distance >= 1.575:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()
                try:
                    distance = abs(float(line))
                    #print(f"Distance: {distance} meters")
                except ValueError:
                    print("Invalid data received")
        send_stop_signal()

        executor.submit(upload_file, data_file)

        time.sleep(wait_time)


except KeyboardInterrupt:
    print("KEYBOARD INTERRUPT; FINALIZING UPLOADS...")
    pyautogui.hotkey('ctrl', 'f12', interval=0.1)
    shutdown_flag.set()
    upload_queue.put(None)

finally:
    uploader_thread.join()
    observer.stop()
    observer.join()
    print("Processes stopped.")




"""Courtesy of ChatGPT"""

import boto3
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import datetime as datetime
import pyautogui
from queue import Queue
import threading


# Initialize S3 client
s3_client = boto3.client('s3')

# Replace with your S3 bucket name
bucket_name = 'spider-videos'

class UploadEventHandler(FileSystemEventHandler):
    def __init__(self, upload_queue) -> None:
        super(UploadEventHandler, self).__init__()
        self.upload_queue = upload_queue

    def on_created(self, event):
        print(f"Detected new file: {event.src_path}")
        if not event.is_directory and event.src_path.lower().endswith(".mp4"):
            self.upload_queue.put(event.src_path)
            print("Adding to queue...")

def upload_worker(upload_queue):
    while True:
        file_path = upload_queue.get()
        if file_path is None:
            break
        upload_file(file_path)
        upload_queue.task_done()


def upload_file(file_path):
    file_name = os.path.basename(file_path)
    print("Verifying file completeness...")
    while not is_file_stable(file_path, wait_time=5, retries=3):
        pass
    print(f'Uploading {file_name} to S3 bucket {bucket_name}...')
    start_time = time.time()
    try:
        s3_client.upload_file(file_path, bucket_name, file_name)
        print(f'{file_name} uploaded.')
        end_time = time.time()
        print(f"Upload took {end_time - start_time} seconds to complete.")
        print("Removing file...")
        os.remove(file_path)
        
    except Exception as e:
        print(f"Error during upload: {e}")

def is_file_stable(file_path, wait_time=2, retries=3):
    last_size = -1

    while True:
        
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            return False
        
        print(f"Checking integrity... current size: {current_size}")
        
        if current_size == last_size:
            retries -= 1
            print(f"File stable; checking {retries} more times.")
        
        last_size = current_size

        if retries <= 0:
            return True
        
        time.sleep(wait_time)
        


# Set up the observer
path = os.path.expanduser('~/Videos')
upload_queue = Queue()
event_handler = UploadEventHandler(upload_queue)
observer = Observer()
observer.schedule(event_handler, path, recursive=False)

time.sleep(5)

# Start the observer
observer.start()

uploader_thread = threading.Thread(target=upload_worker, args=(upload_queue,))
uploader_thread.start()


print(f'Monitoring folder {path} for new video files...')
recording_begin_time = time.time()

# Set video length
video_length = 30 

try:
    while True:
        print(f"Current runtime: {str(datetime.timedelta(seconds=(time.time() - recording_begin_time)))}")
        time.sleep(1)
        pyautogui.keyDown('ctrl')
        time.sleep(0.5)
        pyautogui.keyDown('f11')
        time.sleep(0.5)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('f11')
        time.sleep(video_length)
        pyautogui.keyDown('ctrl')
        time.sleep(0.5)
        pyautogui.keyDown('f12')
        time.sleep(0.5)
        pyautogui.keyUp('ctrl')
        pyautogui.keyUp('f12')
        time.sleep(10)
except KeyboardInterrupt:
    upload_queue.put(None)
    uploader_thread.join()
    observer.stop()
    observer.join()

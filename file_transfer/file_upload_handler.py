"""Courtesy of ChatGPT"""

import boto3
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

# Initialize S3 client
s3_client = boto3.client('s3')

# Replace with your S3 bucket name
bucket_name = 'spider-videos'

class UploadEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        print(f"Detected new file: {event.src_path}")
        if not event.is_directory:
            self.upload_file(event.src_path)

    def upload_file(self, file_path):
        file_name = os.path.basename(file_path)
        print(f'Uploading {file_name} to S3 bucket {bucket_name}...')
        start_time = time.time()
        try:
            s3_client.upload_file(file_path, bucket_name, file_name)
            print(f'{file_name} uploaded.')
            end_time = time.time()
            print(f"Upload took {end_time - start_time} seconds to complete.")
            print()
            
        except Exception as e:
            print(f"Error during upload: {e}")

# Set up the observer
path = os.path.expanduser('~/Documents/spider-recordings')
event_handler = UploadEventHandler()
observer = Observer()
observer.schedule(event_handler, path, recursive=False)

# Start the observer
observer.start()
print('Monitoring folder for new video files...')
try:
    while True:
        # Run indefinitely
        pass
except KeyboardInterrupt:
    observer.stop()

observer.join()

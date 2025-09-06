import os
import time


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
    
def clean_line(line: str):
    i = 0
    for c in line: 
        if not c.isdigit() and not c == '.':
            break
        i += 1
    return line[:i]
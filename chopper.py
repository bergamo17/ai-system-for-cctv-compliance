import subprocess
import os
import time
from config import FRAME_FOLDER, INPUT_FOLDER, FRAME_INTERVAL, FRAME_PER_SECOND

def wait_for_file_complete(path, timeout=30, interval=0.5):
    prev_size = -1
    elapsed = 0
    while elapsed < timeout:
        try:
            curr_size = os.path.getsize(path)
        except FileNotFoundError:
            time.sleep(interval)
            elapsed += interval
            continue

        if curr_size == prev_size and curr_size > 0:
            return True
        
        prev_size = curr_size
        time.sleep(interval)
        elapsed += interval
    
    return False

def chop_video(video_path):
    os.makedirs(FRAME_FOLDER, exist_ok=True)

    print(f"[CAPTURE] Waiting for the new file to finish writing: {video_path}")
    if not wait_for_file_complete(video_path):
        print(f"[ERROR] Waiting file timeout: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_pattern = os.path.join(FRAME_FOLDER, F"{video_name}_frame_%06d.jpg")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={FRAME_PER_SECOND}",
        "-q:v", "2",
        output_pattern
    ]

    print(f"[Capture] Chop the video: {video_path}")
    print(f"[Capture] Output frame to: {FRAME_FOLDER}")

    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg failed:\n{result.stderr.decode()}")

    print(f"[Capture] Capture and Chop done: {video_path}")
    
if __name__ == "__main__":
    print(chop_video(video_path="input/Footage(2 mins).mp4"))
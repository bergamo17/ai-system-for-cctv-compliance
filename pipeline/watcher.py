import os
import time
import shutil
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pipeline.chopper import chop_video
from inference.inference import run_inference
from inference.inference_client import call_inference
from pipeline.summarizer import generate_summary
from config import FRAME_FOLDER, FRAME_INTERVAL, INPUT_FOLDER, OUTPUT_FOLDER

SUPPORTED_FORMATS = (".mp4", ".avi", ".mkv", ".mov")

class VideoHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self._processing = set()
        self._lock = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        filename = os.path.basename(filepath)

        if filename.startswith(".") or filename.startswith("tmp_"):
            return

        if not filename.endswith(SUPPORTED_FORMATS):
            return

        with self._lock:
            if filepath in self._processing:
                return
            self._processing.add(filepath)

        threading.Thread(target=self._handle, args=(filepath, filename), daemon=True).start()

    def on_moved(self, event):
        # ini menangkap event rename dari .tmp_cctv_xxx.mp4 -> cctv_xxx.mp4
        if event.is_directory:
            return

        filepath = event.dest_path
        filename = os.path.basename(filepath)

        if filename.startswith(".") or filename.startswith("tmp_"):
            return

        if not filename.endswith(SUPPORTED_FORMATS):
            return

        with self._lock:
            if filepath in self._processing:
                return
            self._processing.add(filepath)

        threading.Thread(target=self._handle, args=(filepath, filename), daemon=True).start()
        
    def _handle(self, filepath, filename):
        print(f"[Watcher] New video detected: {filename}, waiting for recording to finish...")

        if not self._wait_until_stable(filepath):
            print(f"[Watcher] File lost/failed to get stable: {filename}")
            with self._lock:
                self._processing.discard(filepath)
            return
        
        print(f"[Watcher] File stable, start processing: {filename}")

        success = False

        try:
            frame_dir = chop_video(filepath)
            print(f"[Watcher] Chopping done, starting inference: {filename}")

            video_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(OUTPUT_FOLDER, video_name)

            result = run_inference(frame_folder=frame_dir, output_dir=output_dir)

            if result['violation_log_path'] is None:
                print(f"[Watcher] There is no frame detected for {filename}, skip summary.")
            else: 
                #summary = generate_summary(...)
                print(f"[Watcher] Inference done, generating summary: {filename}")
                summary = generate_summary(result['violation_log_path'], video_name=filename)
                print(f"[Watcher] Summary:\n{summary}")
                success = True
            
        except Exception as e:
            print(f"[Watcher][ERROR] Failed to process {filename}: {e}")

        finally:
            with self._lock:
                self._processing.discard(filepath)
            if success:
                if os.path.exists(filepath):
                    os.remove(filepath)
            else:
                failed_dir = os.path.join(os.path.dirname(filepath), "failed")
                os.makedirs(failed_dir, exist_ok=True)
                shutil.move(filepath, os.path.join(failed_dir, filename))

            if 'frame_dir' in dir() and os.path.exists(frame_dir):
                shutil.rmtree(frame_dir, ignore_errors=True)

    def _wait_until_stable(self, filepath, checks=3, interval=3, max_wait=7200):
        last_size = -1
        stable_count = 0
        waited = 0

        while stable_count < checks and waited < max_wait:
            if not os.path.exists(filepath):
                return False

            size = os.path.getsize(filepath)
            if size == last_size and size > 0:
                stable_count += 1
            else:
                stable_count = 0

            last_size = size
            time.sleep(interval)
            waited += interval

        return stable_count >= checks


def start_watcher():
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    handler = VideoHandler()
    observer = Observer()
    observer.schedule(handler, INPUT_FOLDER, recursive=False)
    observer.start()

    print(f"[Watcher] Monitoring folder: {INPUT_FOLDER}")
    return observer

if __name__ == "__main__":
    observer = start_watcher()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
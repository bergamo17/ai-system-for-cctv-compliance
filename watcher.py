import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from chopper import chop_video
from inference import run_inference
from summarizer import generate_summary
from config import FRAME_FOLDER, FRAME_INTERVAL, INPUT_FOLDER

SUPPORTED_FORMATS = (".mp4", ".avi", ".mkv", ".mov")

class VideoHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self._processing = set()
        self._lock = threading.Lock()

    def on_created(self, event):
        self._handle(event)
        
    #def on_modified(self, event):
    #    self._handle(event)

    def _handle(self, event):
        if event.is_directory:
            return
        
        filepath = event.src_path
        filename = os.path.basename(filepath)

        if not filename.endswith(SUPPORTED_FORMATS):
            print(f"[Watcher] Not the video file, skip: {filename}")
            return
        
        with self._lock:
            if filepath in self._processing:
                print(f"[Watcher] Sudah diproses/sedang diproses, skip: {filename}")
                return
            self._processing.add(filepath)
        
        print(f"[Watcher] New video detected: {filename}")

        time.sleep(2)

        chop_video(filepath)

        print(f"[WATCHER] Chopping done, starting inference: {filename}")

        log_path = run_inference(FRAME_FOLDER)

        print(f"[WATCHER] Inference done, generating LLM summary: {filename}")  # <-- baru
        try:                                                                # <-- baru
            summary = generate_summary(log_path, video_name=filename)  # <-- baru
            print(f"[WATCHER] Summary:\n{summary}")                         # <-- baru
        except Exception as e:                                             # <-- baru
            print(f"[WATCHER][ERROR] Gagal membuat summary: {e}")           # <-- baru


def start_watcher():
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    handler = VideoHandler()
    observer = Observer()
    observer.schedule(handler, INPUT_FOLDER, recursive=False)
    observer.start()

    print(f"[Watcher] Monitoring folder: {INPUT_FOLDER}")
    return observer
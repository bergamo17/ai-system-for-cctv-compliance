import os
import time
import shutil
import logging
import queue
import threading
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from config import FRAME_FOLDER, OUTPUT_FOLDER, FRAME_PER_SECOND
from inference.new_inference import InferenceSession
from pipeline.summarizer import generate_summary

load_dotenv()

RTSP_URL = f"rtsp://{os.getenv('RTSP_USERNAME')}:{os.getenv('RTSP_PASSWORD')}@{os.getenv('RTSP_IP')}:{os.getenv('RTSP_PORT')}/Streaming/Channels/101"
TIMEZONE = ZoneInfo(os.getenv("TIMEZONE"))

logger = logging.getLogger("stream_capture")


def _wait_until_stable(filepath: str, checks: int = 2, interval: float = 0.3, max_wait: float = 10.0) -> bool:
    """Tunggu sampai ukuran file berhenti berubah (selesai ditulis ffmpeg)."""
    last_size = -1
    stable_count = 0
    waited = 0.0

    while stable_count < checks and waited < max_wait:
        if not os.path.exists(filepath):
            return False

        try:
            size = os.path.getsize(filepath)
        except OSError:
            size = -1

        if size == last_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0

        last_size = size
        time.sleep(interval)
        waited += interval

    return stable_count >= checks


class FrameHandler(FileSystemEventHandler):
    """Deteksi frame jpg baru di folder capture sementara, taruh ke antrian
    supaya diproses berurutan (FIFO) oleh 1 worker thread terpisah."""

    def __init__(self, frame_queue: "queue.Queue"):
        super().__init__()
        self._frame_queue = frame_queue

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".jpg"):
            return

        filepath = event.src_path
        if _wait_until_stable(filepath):
            self._frame_queue.put(filepath)
        else:
            logger.warning(f"Frame tidak stabil, dilewati: {filepath}")


def _frame_worker(session: InferenceSession, frame_queue: "queue.Queue"):
    """Tarik path frame dari antrian satu-satu (berurutan) dan proses lewat
    InferenceSession. Berhenti begitu menerima sentinel None."""
    while True:
        frame_path = frame_queue.get()
        if frame_path is None:
            frame_queue.task_done()
            break

        try:
            session.process_frame(frame_path)
        except Exception as e:
            logger.error(f"Gagal memproses frame {frame_path}: {e}")
        finally:
            frame_queue.task_done()


def _run_ffmpeg_capture(cmd: list[str], duration_seconds: int) -> bool:
    """Jalankan ffmpeg non-blocking (Popen), tunggu sampai selesai/timeout.
    Return True kalau capture sukses (returncode 0)."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _, stderr = process.communicate(timeout=duration_seconds + 60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        logger.error("ffmpeg process timed out.")
        return False

    if process.returncode != 0:
        logger.error(f"ffmpeg failed (code {process.returncode}): {stderr[-2000:]}")
        return False

    return True


def process_stream(duration_seconds: int, label: str = "", start_dt: datetime | None = None):
    """Capture RTSP -> tiap frame baru langsung diproses InferenceSession
    (paralel dengan capture yang masih berjalan) -> finalize -> summary.

    start_dt: jam mulai capture, dipakai sebagai basis frame_timestamp di
    log inference (base_dt + frame_count/fps, fps=1 -> +1 detik/frame).
    Kalau tidak diisi (misal dipanggil scheduler.py), default jam sekarang --
    yaitu jam trigger-nya sendiri."""
    if start_dt is None:
        start_dt = datetime.now(TIMEZONE)

    timestamp = start_dt.strftime("%Y%m%d_%H%M%S")
    video_name = f"cctv_{label}_{timestamp}" if label else f"cctv_{timestamp}"

    tmp_frame_dir = Path(FRAME_FOLDER) / f".tmp_{video_name}"
    tmp_frame_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(tmp_frame_dir / "frame_%06d.jpg")
    output_dir = os.path.join(OUTPUT_FOLDER, video_name)

    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-t", str(duration_seconds),
        "-vf", f"fps={FRAME_PER_SECOND}",
        "-q:v", "2",
        output_pattern,
    ]

    session = InferenceSession(output_dir, start_dt=start_dt)
    frame_queue: "queue.Queue" = queue.Queue()

    worker_thread = threading.Thread(target=_frame_worker, args=(session, frame_queue), daemon=True)
    worker_thread.start()

    observer = Observer()
    observer.schedule(FrameHandler(frame_queue), str(tmp_frame_dir), recursive=False)
    observer.start()

    logger.info(f"Start capture+stream [{label}] for {duration_seconds}s -> {tmp_frame_dir}")

    capture_ok = _run_ffmpeg_capture(cmd, duration_seconds)

    observer.stop()
    observer.join()

    frame_queue.put(None)   # sentinel: tidak ada frame baru lagi
    worker_thread.join()    # tunggu backlog frame yang belum sempat diproses

    if not capture_ok:
        logger.error(f"Capture gagal [{label}], tetap finalize sesi untuk frame yang sempat masuk")

    result = session.finalize()

    try:
        if result["violation_log_path"] is None:
            logger.info(f"Tidak ada frame terdeteksi untuk {video_name}, skip summary.")
        else:
            logger.info(f"Inference selesai, generate summary: {video_name}")
            summary = generate_summary(result["violation_log_path"], video_name=video_name, session_start=start_dt.isoformat())
            logger.info(f"Summary:\n{summary}")
    except Exception as e:
        logger.error(f"Gagal generate summary untuk {video_name}: {e}")

    shutil.rmtree(tmp_frame_dir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Capture RTSP stream langsung ke frame (fps=1) tanpa merekam file video utuh, "
                    "tiap frame baru langsung diproses inference (streaming), lalu generate summary."
    )
    parser.add_argument("--duration", type=int, default=300, help="Durasi capture dalam detik (default: 300)")
    parser.add_argument("--label", type=str, default="", help="Label sesi capture (contoh: productivity)")
    parser.add_argument("--start-time", type=str, default=None,
                         help="Jam mulai capture, format HH:MM atau HH:MM:SS (default: jam sekarang). "
                              "Dipakai sebagai basis frame_timestamp di violation log.")
    args = parser.parse_args()

    os.makedirs(FRAME_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    start_dt = None
    if args.start_time:
        now = datetime.now(TIMEZONE)
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                parsed = datetime.strptime(args.start_time, fmt)
                start_dt = now.replace(hour=parsed.hour, minute=parsed.minute,
                                        second=parsed.second, microsecond=0)
                break
            except ValueError:
                continue
        if start_dt is None:
            parser.error("--start-time harus format HH:MM atau HH:MM:SS")

    process_stream(args.duration, args.label, start_dt=start_dt)

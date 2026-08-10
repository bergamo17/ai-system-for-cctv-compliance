import os
import sys
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.date import DateTrigger

load_dotenv()

RTSP_USERNAME = os.getenv("RTSP_USERNAME")
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD")
RTSP_IP = os.getenv("RTSP_IP")
RTSP_PORT = os.getenv("RTSP_PORT")

RTSP_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_IP}:{RTSP_PORT}/Streaming/Channels/101"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "recordings"))
TIMEZONE_NAME = os.getenv("TIMEZONE")
TIMEZONE = ZoneInfo(TIMEZONE_NAME)

DEFAULT_START_HOUR = int(os.getenv("RECORD_START_HOUR", 8))
DEFAULT_START_MINUTE = int(os.getenv("RECORD_START_MINUTE", 0))
DEFAULT_END_HOUR = int(os.getenv("RECORD_END_HOUR", 9))
DEFAULT_END_MINUTE = int(os.getenv("RECORD_END_MINUTE", 0))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rtsp_recorder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ask_int(prompt: str, default: int, min_val: int, max_val: int) -> int:
    while True:
        raw = input(f"{prompt} [default: {default}]: ").strip()

        if raw == "":
            return default

        if not raw.lstrip("-").isdigit():
            print("Invalid input. Please enter a valid integer.")
            continue

        value = int(raw)
        if value < min_val or value > max_val:
            print(f"Invalid input. Please enter an integer between {min_val} and {max_val}")
            continue

        return value

def ask_schedule() -> tuple[int, int, int, int]:
    print("\n=== Set The Recording Schedule ===")
    print("Press Enter without input data to use the default value from .env\n")
 
    start_hour = ask_int("Start hour (0-23)", DEFAULT_START_HOUR, 0, 23)
    start_minute = ask_int("Start minute (0-59)", DEFAULT_START_MINUTE, 0, 59)
    end_hour = ask_int("End hour (0-23)", DEFAULT_END_HOUR, 0, 23)
    end_minute = ask_int("End minute (0-59)", DEFAULT_END_MINUTE, 0, 59)
 
    print()
    return start_hour, start_minute, end_hour, end_minute

def compute_targeted_datetime(start_hour: int, start_minute: int, now: datetime) -> datetime:
    target = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)

    now_without_seconds = now.replace(second=0, microsecond=0)

    if target < now_without_seconds:
        target += timedelta(days=1)

    return target

def compute_duration_seconds(start_hour, start_minute, end_hour, end_minute) -> int:
    dummy_date = datetime(2000, 1, 1)
    start = dummy_date.replace(hour=start_hour, minute=start_minute)
    end = dummy_date.replace(hour=end_hour, minute=end_minute)
 
    if end <= start:
        end += timedelta(days=1)
 
    duration = int((end - start).total_seconds())
 
    if duration <= 0:
        raise ValueError(
            "Recording duration <= 0 detik. Check again the start/end hour."
        )
 
    return duration

def record_stream(duration_seconds: int, scheduler: BlockingScheduler):
    timestamp = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"cctv_{timestamp}.mp4"

    logger.info(f"Start recording -> {output_file}")
    logger.info(f"Target duration: {duration_seconds} seconds")

    cmd = [
        "ffmpeg",
        "-y",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-t", str(duration_seconds),
        "-c:v", "copy",
        "-an",
        str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd, 
            capture_output=True,
            text=True,
            timeout=duration_seconds + 60,
        )

        if result.returncode == 0:
            logger.info(f"Finished recording -> {output_file}")
        else:
            logger.error(f"FFMPEG exited with error (code {result.returncode})")
            logger.error(result.stderr[-2000:])

    except subprocess.TimeoutExpired:
        logger.error("FFMPEG process timed out.")
    except Exception as e:
        logger.error(f"An error occured while recording: {e}")
    finally:
        logger.info("Recording process completed.")
        scheduler.shutdown(wait=False)

def main():
    if not RTSP_IP:
        logger.error("RTSP_IP is not set. Please set it in the .env file.")
        return

    start_hour, start_minute, end_hour, end_minute = ask_schedule()
    duration_seconds = compute_duration_seconds(start_hour, start_minute, end_hour, end_minute)

    now = datetime.now(TIMEZONE)
    target_datetime = compute_targeted_datetime(start_hour, start_minute, now)

    is_today = target_datetime.date() == now.date()
    day_label = "today" if is_today else "tomorrow"

    print(
        f"Confirmed schedule: {start_hour:02d}:{start_minute:02d} - "
        f"{end_hour:02d}:{end_minute:02d} (duration: {duration_seconds // 60 }minute)\n"
    )

    scheduler = BlockingScheduler(timezone=TIMEZONE)

    scheduler.add_job(
        record_stream,
        args=[duration_seconds, scheduler],
        trigger=DateTrigger(
            run_date=target_datetime,
            timezone=TIMEZONE,
        ),
        id="daily_cctv_recording",
        misfire_grace_time=60,
    )

    logger.info(
        f"Scheduler active. Recording will start every day hour "
        f"{start_hour:02d}:{start_minute:02d} until "
        f"{end_hour:02d}:{end_minute:02d} "
        f"({duration_seconds // 60} minute)."
    )
    logger.info("Press Ctrl+C to stop the scheduler.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped. Exiting program.")

    logger.info("Program terminated.")
    sys.exit(0)

if __name__ == "__main__":
    main()

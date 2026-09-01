import os, subprocess, logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from config import INPUT_FOLDER

load_dotenv()

RTSP_URL = f"rtsp://{os.getenv('RTSP_USERNAME')}:{os.getenv('RTSP_PASSWORD')}@{os.getenv('RTSP_IP')}:{os.getenv('RTSP_PORT')}/Streaming/Channels/101"
OUTPUT_DIR = Path(INPUT_FOLDER)
TIMEZONE_NAME = os.getenv("TIMEZONE")
TIMEZONE = ZoneInfo(TIMEZONE_NAME)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rtsp_recorder")

def record_stream(duration_seconds: int, label: str = ""):
    timestamp = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    final_file = OUTPUT_DIR / f"cctv_{label}_{timestamp}.mp4"
    temp_file = OUTPUT_DIR / f".tmp_cctv_{label}_{timestamp}.mp4"

    logger.info(f"Starting recording [{label}] for {duration_seconds} seconds. Output file: {final_file}")

    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-t", str(duration_seconds),
        "-c:v", "copy",
        "-an",
        str(temp_file),
    ]

    try:
        result = subprocess.run(
            cmd, 
            capture_output=True,
            text=True,
            timeout=duration_seconds + 60,
        )

        if result.returncode == 0:
            temp_file.rename(final_file)
            logger.info(f"Recording completed successfully. Output file: {final_file}")
        else:
            logger.error(f"Recording failed with return code {result.returncode}. Error: {result.stderr}")
            if temp_file.exists():
                temp_file.unlink()

    except subprocess.TimeoutExpired:
        logger.error("Recording process timed out.")
        if temp_file.exists():
            temp_file.unlink()

    except Exception as e:
        logger.error(f"An unexpected error occurred during recording: {e}")
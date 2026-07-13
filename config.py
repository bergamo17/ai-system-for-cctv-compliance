import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(BASE_DIR, "input")
FRAME_FOLDER = os.path.join(BASE_DIR, "frames")

FRAME_INTERVAL = 5
FRAME_PER_SECOND = 1 #25 / FRAME_INTERVAL
PRE_VIOLATION_DURATION = 5
POST_VIOLATION_DURATION = 5
#VIOLATION_LOG_PATH   = "output/violation_log.csv"

HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_MODEL = "gpt-5-nano"
MODEL = "moonshotai/Kimi-K2.6:novita"  #"moonshotai/Kimi-K2-Instruct-0905:novita"
SUMMARY_OUTPUT_DIR = "output/summary"

ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
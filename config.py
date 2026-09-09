import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FOLDER = os.path.join(BASE_DIR, "recordings")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
FRAME_FOLDER = os.path.join(BASE_DIR, "frames")

KAGGLE_INFERENCE_URL = os.getenv('KAGGLE_INFERENCE_URL')
DB_PATH = "pipeline/db/Popcorn.db"

FRAME_INTERVAL = 5
FRAME_PER_SECOND = 1 #25 / FRAME_INTERVAL
PRE_VIOLATION_DURATION = 5
POST_VIOLATION_DURATION = 5
#VIOLATION_LOG_PATH   = "output/violation_log.csv"

HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = "moonshotai/Kimi-K2.6:novita"  #"moonshotai/Kimi-K2-Instruct-0905:novita"
SUMMARY_OUTPUT_DIR = "output/summary"

ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

TIMESTAMP_CROP = [
    (6, 47),
    (307, 47),
    (306, 112),
    (4, 115)
]

ACTIVITIES = [
    # 'a person preparing or serving a drink',
    # 'a person standing behind a counter',
    # 'a person talking to a customer at a counter',
    # 'a person cleaning or organizing the counter',
    # 'a person walking behind the counter',
    'a person sitting idle doing nothing',
    'a person using a phone',
    'a person eating food',
    # 'a person lying down sleeping',
    'a person using laptop or computer',
    'a person standing in the room',
    'a person walking in the room',
]

# VIOLATIONS harus selalu subset dari ACTIVITIES
VIOLATIONS = [
    'a person using a phone',
    # 'a person eating food',
    # 'a person lying down sleeping',
    'a person sitting idle doing nothing',
]

# ACTIVE_ACTIVITIES + IDLE_ACTIVITIES harus selalu partisi lengkap dari ACTIVITIES
ACTIVE_ACTIVITIES = [
    # 'a person preparing or serving a drink',
    # 'a person talking to a customer at a counter',
    # 'a person cleaning or organizing the counter',
    # 'a person walking behind the counter',
    'a person using laptop or computer',
]

IDLE_ACTIVITIES = [
    # 'a person standing behind a counter',
    'a person sitting idle doing nothing',
    'a person using a phone',
    'a person eating food',
    'a person standing in the room',
    'a person walking in the room',
    # 'a person lying down sleeping',
]

assert set(VIOLATIONS).issubset(set(ACTIVITIES)), "VIOLATIONS harus subset dari ACTIVITIES"
assert set(ACTIVE_ACTIVITIES) | set(IDLE_ACTIVITIES) == set(ACTIVITIES), \
    "ACTIVE_ACTIVITIES + IDLE_ACTIVITIES harus mencakup semua ACTIVITIES tanpa sisa"
assert set(ACTIVE_ACTIVITIES) & set(IDLE_ACTIVITIES) == set(), \
    "ACTIVE_ACTIVITIES dan IDLE_ACTIVITIES tidak boleh overlap"

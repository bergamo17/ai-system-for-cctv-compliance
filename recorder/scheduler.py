import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from recorder.recorder_core import record_stream, TIMEZONE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("scheduler")

ONE_MINUTE = 60

TRIGGERS = [
    (8, 0, "attendance_sop"),
    (10, 0, "productivity"),
    (12, 0, "peak_productivity"),
    (13, 0, "peak_productivity"),
    (15, 30, "productivity"),
    (18, 0, "peak_productivity"),
    (19, 0, "peak_productivity"),
    (22, 0, "attendance_sop"),
]

def main():
    scheduler = BlockingScheduler(timezone=TIMEZONE)

    for hour, minute, label in TRIGGERS:
        scheduler.add_job(
            record_stream,
            args=[ONE_MINUTE, label],
            trigger=CronTrigger(hour=hour, minute=minute, second=0, timezone=TIMEZONE),
            id=f"record_{label}_{hour:02d}{minute:02d}",
            misfire_grace_time=120,
        )
        logger.info(f"Registered trigger {hour:02d}:{minute:02d} -> {label}")

    logger.info("Scheduler started, running daily. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped. Exiting program.")


if __name__ == "__main__":
    main()
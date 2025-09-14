from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


INCOMING = Path("/home/jingyi/data/incoming")


def run_kedro():
    logger.info("Trigger kedro run...")
    try:
        subprocess.run(["kedro", "run"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"kedro run failed: {e}")


class Handler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() in {".csv", ".parquet"}:
            logger.info(f"Detected change: {p}")
            run_kedro()


def main(once: bool = False):
    INCOMING.mkdir(parents=True, exist_ok=True)
    if once:
        run_kedro()
        return
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, str(INCOMING), recursive=False)
    observer.start()
    logger.info(f"Watching {INCOMING} for new CSV/Parquet files...")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    main(once=args.once)


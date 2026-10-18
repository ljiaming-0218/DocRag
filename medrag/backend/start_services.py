"""Start the API and research worker together, locally or inside Docker."""

import argparse
import logging
from pathlib import Path
import signal
import subprocess
import sys
import time


logger = logging.getLogger(__name__)


def serve(host="127.0.0.1", port=8000):
    backend_dir = Path(__file__).resolve().parent
    children = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous_handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        commands = (
            ("api", [sys.executable, "-m", "uvicorn", "main:app",
                     "--host", host, "--port", str(port)]),
            ("research_worker", [sys.executable, "-m", "harness.research_worker"]),
        )
        for name, command in commands:
            process = subprocess.Popen(command, cwd=backend_dir)
            children.append((name, process))
            logger.info("service_started name=%s pid=%s", name, process.pid)
        while not stopping:
            for name, process in children:
                code = process.poll()
                if code is not None:
                    logger.error("service_exited name=%s exit_code=%s", name, code)
                    return code if code > 0 else 1
            time.sleep(0.5)
        return 0
    finally:
        for _, process in children:
            if process.poll() is None:
                process.terminate()
        for _, process in children:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    sys.exit(serve(args.host, args.port))

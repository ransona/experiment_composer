import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def run_job(job, base_dir):
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = job["log_path"]

    def log(msg):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"START {datetime.now().isoformat()}")
    log(f"JOB {json.dumps(job)}")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "export_from_template.py"),
        "--template", job["template"],
        "--expID", job["expID"],
        "--userID", job["userID"],
        "--start", str(job["start"]),
        "--stop", str(job["stop"]),
        "--fps", str(job.get("sample_fps", job.get("fps", 0))),
        "--play-fps", str(job.get("play_fps", job.get("fps", job.get("sample_fps", 0)))),
        "--out", job["out"],
        "--log", log_path,
    ]
    try:
        log(f"RUN {' '.join(cmd)}")
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            log(f"ERROR export failed with code {proc.returncode}")
    except Exception as e:
        log(f"ERROR {e}")
    log("DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--base-dir", required=True)
    args = parser.parse_args()

    base_dir = os.path.join(args.base_dir, args.user)
    queue_path = os.path.join(base_dir, "queue.jsonl")
    idx_path = os.path.join(base_dir, "queue_index.txt")
    os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists(queue_path):
        with open(queue_path, "w", encoding="utf-8"):
            pass
    if not os.path.exists(idx_path):
        with open(idx_path, "w", encoding="utf-8") as f:
            f.write("0")

    heartbeat = os.path.join(base_dir, "worker_heartbeat.log")
    def hb(msg):
        with open(heartbeat, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    hb(f"WORKER START {datetime.now().isoformat()}")
    while True:
        hb(f"SCAN {datetime.now().isoformat()}")
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                start_idx = int(f.read().strip() or "0")
        except Exception:
            start_idx = 0

        try:
            with open(queue_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            lines = []
        if len(lines) == 1 and "\\n" in lines[0]:
            lines = [ln for ln in lines[0].split("\\n") if ln.strip()]

        if start_idx < len(lines):
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                except Exception:
                    continue
                hb(f"JOB START {job.get('job_id','?')}")
                print(f"Starting job {job.get('job_id','?')} -> {job.get('out','')}", flush=True)
                run_job(job, base_dir)
                start_idx = i + 1
                with open(idx_path, "w", encoding="utf-8") as f:
                    f.write(str(start_idx))
                hb(f"JOB DONE {job.get('job_id','?')}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()

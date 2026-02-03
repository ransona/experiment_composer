import os
import sys
import time

base = "/data/common/composer/temp_tmux"
user = os.environ.get("USER", "unknown")
base_dir = os.path.join(base, user)
os.makedirs(base_dir, exist_ok=True)
log_path = os.path.join(base_dir, "tmux_test.log")

with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"START {time.time()} pid={os.getpid()}\n")

for i in range(5):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"TICK {i} {time.time()}\n")
    time.sleep(1)

with open(log_path, "a", encoding="utf-8") as f:
    f.write("DONE\n")

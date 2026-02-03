#!/usr/bin/env bash
set -euo pipefail
SESSION="composer"
CMD="$(command -v python) /home/adamranson/code/experiment_composer/test_tmux_run.py"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n tmux_test "$CMD"
else
  tmux new-session -d -s "$SESSION" "$CMD"
fi
printf 'Launched in tmux session %s\n' "$SESSION"

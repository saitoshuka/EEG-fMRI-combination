#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sudaxin/projects/paired_data"
SYNC="${ROOT}/fmri_foundation_workspace/scripts/sync_thing_eeg_to_wsl.sh"
LOG_DIR="${ROOT}/fmri_foundation_workspace/results/eeg_image_bridge/logs"
PID_FILE="${LOG_DIR}/rsync_preprocessed_data_250hz_to_wsl.pid"
NOHUP_LOG="${LOG_DIR}/rsync_preprocessed_data_250hz_to_wsl.nohup"
TMUX_SESSION="thing_eeg_sync"

mkdir -p "${LOG_DIR}"
chmod +x "${SYNC}"

if pgrep -f "${SYNC}" >/dev/null; then
  echo "sync already running"
  pgrep -af "${SYNC}"
  exit 0
fi

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${TMUX_SESSION}"
    tmux list-panes -t "${TMUX_SESSION}" -F "#{pane_pid} #{pane_current_command}"
    exit 0
  fi

  tmux new-session -d -s "${TMUX_SESSION}" "${SYNC}"
  pid="$(tmux display-message -p -t "${TMUX_SESSION}" "#{pane_pid}")"
  echo "${pid}" >"${PID_FILE}"
  echo "started sync in tmux session=${TMUX_SESSION} pid=${pid}"
  tmux list-panes -t "${TMUX_SESSION}" -F "#{pane_pid} #{pane_current_command}"
  exit 0
fi

setsid -f bash -c "exec '${SYNC}' >'${NOHUP_LOG}' 2>&1"
sleep 2
pid="$(pgrep -fo "${SYNC}" || true)"
echo "${pid}" >"${PID_FILE}"

echo "started sync pid=${pid}"
if [[ -n "${pid}" ]]; then
  ps -p "${pid}" -o pid,stat,cmd
fi

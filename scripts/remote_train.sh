#!/usr/bin/env bash
# Run a training command on a remote GPU host as if it were local.
#
# The experiment skill builds `timeout ... {train_command} > train.log 2>&1 &` and then
# waits on it, so anything substituted for {train_command} must behave like a local
# training process: stream to stdout, block until training ends, and exit with the
# training command's real status. This wrapper does that while the work runs elsewhere.
#
#   remote_train.sh --host H --workdir W [--gpu N] [--env-python P] [--sync DIR] -- CMD...
#
#   --host        ssh destination (an ~/.ssh/config alias is fine)
#   --workdir     absolute path on the host where CMD runs (a leading ~ is resolved)
#   --gpu         value for CUDA_VISIBLE_DEVICES (omit to leave the host's default)
#   --env-python  interpreter on the host; replaces a leading `python`/`python3` in CMD
#   --sync        local directory mirrored into --workdir before launching (rsync
#                 --delete-after, so a second --sync erases the first one's files)
#
set -uo pipefail

HOST="" WORKDIR="" GPU="" ENV_PYTHON="" ; SYNC_DIRS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host)       HOST="$2"; shift 2 ;;
        --workdir)    WORKDIR="$2"; shift 2 ;;
        --gpu)        GPU="$2"; shift 2 ;;
        --env-python) ENV_PYTHON="$2"; shift 2 ;;
        --sync)       SYNC_DIRS+=("$2"); shift 2 ;;
        --)           shift; break ;;
        *)            echo "remote_train.sh: unknown option '$1'" >&2; exit 2 ;;
    esac
done
TRAIN_CMD=("$@")

[ -n "$HOST" ]    || { echo "remote_train.sh: --host is required" >&2; exit 2; }
[ -n "$WORKDIR" ] || { echo "remote_train.sh: --workdir is required" >&2; exit 2; }

# Remote paths are single-quoted below (so spaces survive), which also blocks tilde
# expansion — resolve a leading ~ here so --workdir '~/runs' works as it would in ssh.
case "$WORKDIR" in
    '~'|'~/'*)
        REMOTE_HOME=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" 'printf %s "$HOME"' 2>/dev/null)
        [ -n "$REMOTE_HOME" ] || { echo "remote_train.sh: cannot resolve ~ on $HOST" >&2; exit 1; }
        WORKDIR="$REMOTE_HOME${WORKDIR#\~}"
        ;;
esac
[ ${#TRAIN_CMD[@]} -gt 0 ] || { echo "remote_train.sh: no training command after --" >&2; exit 2; }

# Reuse one authenticated connection across the ssh/rsync calls below; Phase 7 runs many
# experiments in parallel. Short fixed dir, not $TMPDIR: the socket path caps near 104
# bytes, and a long $TMPDIR plus ssh's %C hash overflows it and silently kills reuse.
CTL_DIR="${HOME}/.ssh/cm"
mkdir -p "$CTL_DIR" 2>/dev/null && chmod 700 "$CTL_DIR" 2>/dev/null
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
if [ -d "$CTL_DIR" ] && [ ${#CTL_DIR} -lt 70 ]; then
    SSH+=(-o ControlMaster=auto -o ControlPath="$CTL_DIR/%C" -o ControlPersist=60)
fi

# `$$` in the hash makes the name unique per invocation, so parallel experiments never
# collide and the trap kills exactly this run.
SESSION="mlopt-$(printf '%s' "$WORKDIR ${TRAIN_CMD[*]} $GPU $$" | cksum | cut -d' ' -f1)"

cleanup() {
    local sig="$1"
    echo "" >&2
    echo "remote_train.sh: received $sig — terminating remote session $SESSION on $HOST" >&2
    # Drop the local log-follower first so it cannot hold the script open.
    [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
    "${SSH[@]}" "$HOST" "tmux kill-session -t '$SESSION' 2>/dev/null; \
        pkill -f 'MLOPT_SESSION=$SESSION' 2>/dev/null; true" >/dev/null 2>&1 || true
    "${SSH[@]}" -O exit "$HOST" >/dev/null 2>&1 || true
    # 128+15 (SIGTERM) matches what the plugin would see from a locally-killed process.
    exit $((128 + $([ "$sig" = "SIGINT" ] && echo 2 || echo 15)))
}
trap 'cleanup SIGTERM' TERM
trap 'cleanup SIGINT'  INT

if ! "${SSH[@]}" "$HOST" "mkdir -p '$WORKDIR'" 2>/dev/null; then
    echo "remote_train.sh: cannot reach $HOST or create $WORKDIR" >&2
    exit 1
fi

for dir in ${SYNC_DIRS+"${SYNC_DIRS[@]}"}; do
    [ -d "$dir" ] || { echo "remote_train.sh: --sync '$dir' is not a directory" >&2; exit 2; }
    # Trailing slash: copy the directory's contents, not the directory itself.
    if ! rsync -az --delete-after \
            --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
            -e "ssh -o BatchMode=yes -o ControlPath='$CTL_DIR/%C'" \
            "${dir%/}/" "$HOST:$WORKDIR/" 2>&1; then
        echo "remote_train.sh: rsync of '$dir' failed" >&2
        exit 1
    fi
done

# The plugin composes commands against the *local* environment, whose `python` is the
# wrong interpreter on a GPU host.
if [ -n "$ENV_PYTHON" ]; then
    case "${TRAIN_CMD[0]}" in
        python|python3) TRAIN_CMD[0]="$ENV_PYTHON" ;;
    esac
fi

# Quote every argument so the remote shell receives them exactly as given.
REMOTE_CMD=""
for arg in "${TRAIN_CMD[@]}"; do
    REMOTE_CMD+="$(printf '%q ' "$arg")"
done

REMOTE_LOG="$WORKDIR/.mlopt-$SESSION.log"
REMOTE_RC="$WORKDIR/.mlopt-$SESSION.rc"
GPU_PREFIX=""
[ -n "$GPU" ] && GPU_PREFIX="CUDA_VISIBLE_DEVICES=$GPU "

# Detached launch. The exit code lands in a file because tmux's own status says only
# whether the *session* ended, not how training exited.
LAUNCH="cd '$WORKDIR' && rm -f '$REMOTE_LOG' '$REMOTE_RC' && \
tmux new -d -s '$SESSION' \"MLOPT_SESSION=$SESSION ${GPU_PREFIX}${REMOTE_CMD} > '$REMOTE_LOG' 2>&1; echo \\\$? > '$REMOTE_RC'\""

if ! "${SSH[@]}" "$HOST" "$LAUNCH" 2>&1; then
    echo "remote_train.sh: failed to launch tmux session on $HOST" >&2
    exit 1
fi

echo "remote_train.sh: launched $SESSION on $HOST${GPU:+ (GPU $GPU)}" >&2

# Follow the remote log until the run records its exit code (`tail -F` tolerates the file
# not existing yet; completion is signalled by the .rc file, so the poll loop waiting on
# it also kills the tail).
#
# Backgrounded and waited on, which is load-bearing: bash defers traps during a foreground
# command, so tailing in the foreground would swallow the plugin's timeout SIGTERM until
# training ended on its own. `wait` is interruptible, so the trap fires at once.
"${SSH[@]}" "$HOST" "tail -F -n +1 '$REMOTE_LOG' 2>/dev/null & TAIL=\$!; \
    while [ ! -f '$REMOTE_RC' ]; do sleep 2; done; \
    sleep 2; kill \$TAIL 2>/dev/null; true" &
TAIL_PID=$!
wait "$TAIL_PID"

RC=$("${SSH[@]}" "$HOST" "cat '$REMOTE_RC' 2>/dev/null" | tr -d '[:space:]')
"${SSH[@]}" "$HOST" "tmux kill-session -t '$SESSION' 2>/dev/null; true" >/dev/null 2>&1 || true
"${SSH[@]}" -O exit "$HOST" >/dev/null 2>&1 || true

trap - TERM INT
case "$RC" in
    ''|*[!0-9]*) echo "remote_train.sh: no exit status recorded on $HOST" >&2; exit 1 ;;
    *)           exit "$RC" ;;
esac

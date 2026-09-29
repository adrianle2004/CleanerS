#!/usr/bin/env bash
# reconstruction_GT/guard.sh -- run a long job so a closed lid cannot lose it.
#
#   ./reconstruction_GT/guard.sh <tag> <command...>
#
#   ./reconstruction_GT/guard.sh sweep8 \
#       python -u -m reconstruction_GT.sweep_eval captures/room08 --every 5
#
# While it runs:
#
#   touch  outputs/guard/<tag>.pause     pause the job (SIGSTOP)
#   rm     outputs/guard/<tag>.pause     let it carry on (SIGCONT)
#   touch  outputs/guard/<tag>.stop      stop for good, no restart
#   tail -f outputs/guard/<tag>.log      watch it
#
# *** WHY ***
#
# The long jobs here take tens of minutes on the GPU. Suspending the machine
# mid-run either freezes them or leaves the CUDA context wedged, and a job that
# has to start from nothing after every interruption never finishes on a
# laptop. So this does two things:
#
#   1. it restarts the command if its OUTPUT goes quiet, which is the symptom
#      of a wedged GPU call -- a job that is merely slow keeps printing;
#   2. it measures that quiet in /proc/uptime, which does NOT advance while the
#      machine is asleep. Close the lid for an hour and the guard sees no stall
#      at all, because from the job's point of view no time passed.
#
# Restarting only helps if the job can pick up where it left off, so the jobs
# it wraps skip work that is already on disk -- `run_inference` leaves existing
# predictions alone, `sweep_eval` keeps its CSV and skips frames already in it.
# The guard is the retry; resumability is theirs.

set -u

STALL_S="${STALL_S:-300}"       # quiet for this long (awake) = wedged
MAX_TRIES="${MAX_TRIES:-5}"
POLL_S="${POLL_S:-10}"

if [ $# -lt 2 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

TAG="$1"; shift
DIR="outputs/guard"
mkdir -p "$DIR"
LOG="$DIR/$TAG.log"
PAUSE="$DIR/$TAG.pause"
STOP="$DIR/$TAG.stop"
STATE="$DIR/$TAG.state"
rm -f "$STOP"

uptime_s() { awk '{printf "%d", $1}' /proc/uptime; }

say() { printf '[guard %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

say "starting: $*"
say "log $LOG   pause: touch $PAUSE   stop: touch $STOP"

try=0
while [ "$try" -lt "$MAX_TRIES" ]; do
    try=$((try + 1))
    [ -f "$STOP" ] && { say "stop file present, not starting"; exit 0; }

    setsid "$@" >>"$LOG" 2>&1 &
    child=$!
    pgid=$(ps -o pgid= -p "$child" 2>/dev/null | tr -d ' ')
    printf 'tag=%s try=%d pid=%d pgid=%s started=%s\n' \
        "$TAG" "$try" "$child" "${pgid:-?}" "$(date -Is)" >"$STATE"
    say "attempt $try of $MAX_TRIES, pid $child"

    last_size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    last_move=$(uptime_s)
    paused=0

    while kill -0 "$child" 2>/dev/null; do
        sleep "$POLL_S"

        if [ -f "$STOP" ]; then
            say "stop requested, terminating"
            kill -TERM -"${pgid:-$child}" 2>/dev/null || kill -TERM "$child" 2>/dev/null
            wait "$child" 2>/dev/null
            exit 0
        fi

        if [ -f "$PAUSE" ] && [ "$paused" -eq 0 ]; then
            kill -STOP -"${pgid:-$child}" 2>/dev/null || kill -STOP "$child" 2>/dev/null
            paused=1
            say "paused (rm $PAUSE to continue)"
        elif [ ! -f "$PAUSE" ] && [ "$paused" -eq 1 ]; then
            kill -CONT -"${pgid:-$child}" 2>/dev/null || kill -CONT "$child" 2>/dev/null
            paused=0
            last_move=$(uptime_s)          # do not blame the pause for the quiet
            say "resumed"
        fi
        [ "$paused" -eq 1 ] && continue

        size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
        if [ "$size" != "$last_size" ]; then
            last_size="$size"
            last_move=$(uptime_s)
            continue
        fi
        # awake-only seconds since the log last grew: sleeping the laptop does
        # not advance this, so a closed lid never looks like a stall
        quiet=$(( $(uptime_s) - last_move ))
        if [ "$quiet" -ge "$STALL_S" ]; then
            say "no output for ${quiet}s of awake time -- restarting"
            kill -TERM -"${pgid:-$child}" 2>/dev/null || kill -TERM "$child" 2>/dev/null
            sleep 5
            kill -KILL -"${pgid:-$child}" 2>/dev/null || true
            break
        fi
    done

    if wait "$child" 2>/dev/null; then
        say "finished cleanly on attempt $try"
        printf 'tag=%s done=%s attempts=%d\n' "$TAG" "$(date -Is)" "$try" >"$STATE"
        exit 0
    fi
    rc=$?
    if [ -f "$STOP" ]; then say "stopped"; exit 0; fi
    say "exited with status $rc; retrying in 10s (the job skips finished work)"
    sleep 10
done

say "gave up after $MAX_TRIES attempts"
exit 1

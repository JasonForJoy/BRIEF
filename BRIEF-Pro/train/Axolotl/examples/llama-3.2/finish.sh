#!/bin/bash

# USERNAME=jcgu   # your username here
# PATTERN=wandb-core
# pgrep -u $USERNAME -f "^$PATTERN" | while read PID; do
#     echo "Killing process ID $PID"
#     kill $PID
# done
ps aux | grep "axo/lib/python3.10/site-packages/wandb/bin/wandb" | grep -v "grep" | awk '{print $2}' | while read PID; do
    echo "Killing process ID $PID"
    kill $PID
    sleep 10  # Give it some time to terminate

    # If the process is still running, forcefully kill it
    if ps -p $PID > /dev/null; then
        echo "Graceful shutdown failed. Forcing termination for process $PID..."
        kill -9 $PID
    else
        echo "Process $PID terminated gracefully."
    fi
done

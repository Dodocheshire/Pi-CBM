#!/usr/bin/env bash
# Source this file on ZJU-2 before launching experiments. These libraries match
# the already loaded kernel module; no system-wide driver changes are needed.
export LD_LIBRARY_PATH="$HOME/.local/share/nvidia-userspace/580.173.02/extracted/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/miniconda3/envs/pi-cbm/bin:$PATH"
export PYTHONUNBUFFERED=1

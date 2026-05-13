#!/bin/bash

# 1. Create the system-level fix for NVIDIA profiling
sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiler.conf'

# 2. (Optional) Install other useful tools for CUDA dev
sudo apt-get update
sudo apt-get install -y nvtop htop

echo "--------------------------------------------------------"
echo "System configured! Please RESTART the VM in Lambda now."
echo "--------------------------------------------------------"
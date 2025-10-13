import time
import subprocess
from functools import wraps
import os
import fcntl
"""
Free:
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

In use:
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A    149287      C   ...conda3/envs/multidoc3/bin/python3.9      60280MiB |
+-----------------------------------------------------------------------------------------+
"""

# Function to check GPU usage for specific GPUs (0, 1, 2, 3)
def is_gpu_free(list_of_gpus=[0, 1, 2, 3]):
    try:
        for gpu in list_of_gpus:
            gpu = str(gpu)
            result = subprocess.run(['nvidia-smi', '-i', gpu], stdout=subprocess.PIPE, text=True)
            output = result.stdout
            if "No running processes found" in output:
                continue
            else:
                return False
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def wait_until_free_do(func, gpu_lists=[0, 1, 2, 3],lock_file="gpu_check_lock_file",check_freq=600):
    @wraps(func)
    def wait_func(*args, **kwargs):
        print(f"Wait until free for {gpu_lists}...")
        while True:
            with open(lock_file,"w") as f:
                try:
                    fcntl.flock(f,fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if is_gpu_free(gpu_lists):
                        print(f"Specified GPUs {gpu_lists} are free. Running the Python script...")
                        return func(*args, **kwargs)
                    else:
                        print(f"One or more of the specified GPUs {gpu_lists} are in use. Checking again in {check_freq} seconds...")
                        time.sleep(check_freq)
                except IOError as E:
                    print(f"The lock file is in use. {E} Checking again in {check_freq} seconds...")
                    time.sleep(check_freq)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
    return wait_func


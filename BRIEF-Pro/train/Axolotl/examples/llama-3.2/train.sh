#!/bin/bash

# WANDB ENVIRONMENT VARIABLES
# Set the WANDB_DIR environment variable to specify where W&B stores generated files

export WANDB_DIR="/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/WandbFiles"
export WANDB_ARTIFACT_DIR="/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/WandbFiles/artifacts"
export TMPDIR="/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/WandbFiles/tmp"


# Echo the paths to confirm they are set correctly
echo "WANDB_DIR is set to: $WANDB_DIR"
echo "WANDB_ARTIFACT_DIR is set to: $WANDB_ARTIFACT_DIR"
echo "TMPDIR is set to: $TMPDIR"

axolotl_cfgs=(
    "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-Brief_Pro_UserControl.yml"
)

# Function to check if a GPU is free
function is_gpu_free() {
  gpu_id="$1"
  output=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
  used_memory=${output//MiB/}  # Remove trailing " MiB"

  # Set a reasonable free memory threshold (adjust based on your needs)
  free_memory_threshold=2048  # MiB

  if [[ "$used_memory" -lt "$free_memory_threshold" ]]; then
    return 0  # GPU is free
  else
    return 1  # GPU is not free
  fi
}
ini_interval=0
echo "Sleeping for $ini_interval seconds after initialization..."
sleep "$ini_interval"

# List of GPUs to check (modify as needed)
# gpus=(2 3 4 5)
gpus=(1 2)

# Interval between checks (adjust in seconds)
check_interval=480

while :; do
  # Check if all specified GPUs are free
  all_free=true
  for gpu in "${gpus[@]}"; do
    if ! is_gpu_free "$gpu"; then
      all_free=false
      break
    fi
  done

  if [[ $all_free == true ]]; then
    # Run the Python script in the background and capture its output
    # python examples/llama-3.2/test.py 2>&1 | while IFS= read -r line; do
    for cfg in "${axolotl_cfgs[@]}"; do
        echo "Running $cfg..."
        # accelerate launch --config_file /home/jcgu/.cache/huggingface/accelerate/default_config_2.yaml -m axolotl.cli.train "$cfg" 2>&1 | while IFS= read -r line; do
        
        CUDA_VISIBLE_DEVICES=1,2 accelerate launch --main_process_port 26345 -m axolotl.cli.train "$cfg" 2>&1 | while IFS= read -r line; do


            echo "$line"  # Print the line for monitoring

            # Check if "Training Finished" is in the line
            if echo "$line" | grep -q "Post train stuff complete"; then
                echo "Detected 'Training Finished' in output. Waiting for 3 minutes..."
                sleep 180  # Wait for 5 minutes (300 seconds)

                # # Call the finish.sh script
                echo "Executing finish.sh..."
                # bash examples/llama-3.2/finish.sh
                bash ./finish.sh
                break  # Exit the loop once finish.sh is called
            fi
        done
        sleep 60
    done

    # Optional: Exit after running the code (uncomment if desired)
    exit 0
  fi
  echo "Not free yet. Sleeping for $check_interval seconds..."
  sleep "$check_interval"
done



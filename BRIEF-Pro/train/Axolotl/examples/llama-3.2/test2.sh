#!/bin/bash

# WANDB ENVIRONMENT VARIABLES
# Set the WANDB_DIR environment variable to specify where W&B stores generated files
export WANDB_DIR="/local2/diwu/lyk/WandbFiles"

# Set the WANDB_ARTIFACT_DIR environment variable to control where artifacts are stored
export WANDB_ARTIFACT_DIR="./artifacts"
export TMPDIR="/local2/diwu/lyk/WandbFiles/tmp"


# Echo the paths to confirm they are set correctly
echo "WANDB_DIR is set to: $WANDB_DIR"
echo "WANDB_ARTIFACT_DIR is set to: $WANDB_ARTIFACT_DIR"
echo "TMPDIR is set to: $TMPDIR"

axolotl_cfgs=(
    # "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-large_strict.yml"
    # "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-oracle_supporting.yml"
    # "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-small_strict.yml"
    # "examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0002-prop_eps0_005_supporting.yml"
    # "examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep1-lr0_0001-oracle_supporting.yml"
    # "examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep2-lr0_0001-oracle_supporting.yml"
    # "examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep2-lr5e-05-oracle_supporting.yml"
    # "examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0002-oracle_supporting.yml"
    # "examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep2-lr0_0002-oracle_supporting.yml"
    # "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-small_cont.yml"
    # "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-cont.yml"
    # "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont.yml"
    # "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-simplemix.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-1.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-3.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-5.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-shorter.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-7.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-9.yml"
    "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki.yml"
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

# List of GPUs to check (modify as needed)
# gpus=(2 3 4 5)
gpus=(4 5)

# Interval between checks (adjust in seconds)
check_interval=15

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
        accelerate launch --config_file /local2/diwu/lyk/MultiHopRAG/tempconfig.yaml --main_process_port 29513 -m axolotl.cli.train "$cfg" 2>&1 | while IFS= read -r line; do
            echo "$line"  # Print the line for monitoring

            # Check if "Training Finished" is in the line
            if echo "$line" | grep -q "Post train stuff complete"; then
                echo "Detected 'Training Finished' in output. Waiting for 3 minutes..."
                sleep 180  # Wait for 5 minutes (300 seconds)

                # # Call the finish.sh script
                echo "Executing finish.sh..."
                bash examples/llama-3.2/finish.sh
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
# axolotl_cfgs=(
#     "examples/llama-3.2/series/lora-3b-ep3.yml"
#     "examples/llama-3.2/series/lora-3b-ep3-lr5e-5.yml"
#     "examples/llama-3.2/series/instruct-lora-3b.yml"
#     "examples/llama-3.2/series/instruct-lora-3b-ep3.yml"
#     "examples/llama-3.2/series/instruct-lora-3b-ep3-lr5e-5.yml"
# )
# axolotl_cfgs=(
#     # "examples/llama-3.2/series/lora-3b-ep3-lr2e-4.yml"
#     # "examples/llama-3.2/series/lora-3b-ep3-lr3e-4.yml"
#     "examples/llama-3.2/series/instruct-lora-3b-ep3-lr2e-4.yml"
#     "examples/llama-3.2/series/instruct-lora-3b-ep3-lr3e-4.yml"
    # "examples/llama-3.2/series/datasets/lora-3b-ep3-lr2e-4-support.yml"
    # "examples/llama-3.2/series/datasets/lora-3b-ep3-lr2e-4-top5.yml"
    # "examples/llama-3.2/series/datasets/instruct-lora-3b-ep3-lr2e-4-top5.yml"
    # "examples/llama-3.2/series/datasets/instruct-lora-3b-ep3-lr2e-4-top2.yml"
    # "examples/llama-3.2/series/datasets/instruct-lora-3b-ep3-lr2e-4-top5.yml"
    # "examples/llama-3.2/series/instruct-lora-3b-ep3-lr2e-4.yml"
    # "examples/llama-3.2/series/datasets/instruct-lora-3b-ep3-lr2e-4-support.yml"


    # "examples/llama-3.2/series/datasets/instruct-lora-3b-ep3-lr2e-4-prune-supp.yml"
    # "examples/llama-3.2/series/datasets/instruct-lora-3b-ep3-lr2e-4-prop-supp.yml" # For pruned prop, with eps = 0.5%
# )


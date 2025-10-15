#!/bin/bash

# Activate the conda environment, if needed
# conda activate your_conda_environment_name

# Wait for 40 minutes
# sleep 2400

# Run the first training command
accelerate launch -m axolotl.cli.train examples/llama-3.2/lora-3b-hyper1.yml
# Run the second training command once the first one is finished
# accelerate launch -m axolotl.cli.train examples/llama-3.2/lora-3b-hyper2.yml

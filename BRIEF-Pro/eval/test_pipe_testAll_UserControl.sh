


source ~/.bashrc


ALLOW_GPUS="5,6"
ALLOW_NUM_GPU="2"
SETTINGS=(
    "Brief-Pro"
    )
MERGE_PATHS=(
    "./brief-pro_checkpoint"
    )

for i in "${!SETTINGS[@]}"; do
    SETTING="${SETTINGS[$i]}"
    MERGE_PATH="${MERGE_PATHS[$i]}"


    # DO THE COMPRESSION FOR LONGBENCH
    conda activate multidoc_vllm

    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python ./vllm_compression_longbench_testAll_UserControl.py --name "${SETTING}" --model_path "${MERGE_PATH}" --parallel_size "${ALLOW_NUM_GPU}"



    # DO THE READING FOR LONGBENCH
    conda activate longbench

    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python ./pred_testAll.py --model "llama3.1-8b-instruct-128k" --e --setting "${SETTING}"
    
    
    # EVAL
    python ./eval.py --model "llama3.1-8b-instruct-128k_${SETTING}"


done


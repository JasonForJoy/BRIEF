
source /data2/junyizhang/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc


ALLOW_GPUS="2,3"
ALLOW_NUM_GPU="2"
SETTINGS=(
    # "My_Axolotl_Finetuned_MuSiqQue_x4_UserControl_Final_10sentences_testAll"
    # "My_Axolotl_Finetuned_MuSiqQue_x4_UserControl_Final_Auto"
    "noCompress_testAll"
    )

    

for i in "${!SETTINGS[@]}"; do
    SETTING="${SETTINGS[$i]}"
    # MERGE_PATH="${MERGE_PATHS[$i]}"



    # DO THE COMPRESSION FOR LONGBENCH
    
    # conda activate multidoc_vllm
    
    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_testAll_UserControl.py --name "${SETTING}" --model_path "${MERGE_PATH}" --parallel_size "${ALLOW_NUM_GPU}"



    # DO THE READING FOR LONGBENCH
    conda activate longbench

    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/pred_noCompress_testAll.py --model "llama3.1-8b-instruct-128k" --e --setting "${SETTING}"
    
    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/pred_noCompress_testFanOutQA.py --model "llama3.1-8b-instruct-128k" --e --setting "${SETTING}"


    python /data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/eval_FanOutQA.py --model "llama3.1-8b-instruct-128k_${SETTING}"


done



source /data2/junyizhang/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc


ALLOW_GPUS="6,7"
ALLOW_NUM_GPU="2"
SETTINGS=(
    "My_Axolotl_Finetuned_MuSiqQue_x4_UserControl_version5_2_continueFinetune_checkpoint1090_Auto"
    )
MERGE_PATHS=(
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/lora-out-ins-epoch3-lr0_0001-hotpotOracle_10longer-merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/llama3_full_merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/llama3_full_merged_checkpoint-5624"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/MuSiqQue_x4_llama3_Final_full_merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/lora-out-ins-epoch3-lr0_0001-sent_wikifull_8longer_fixed_mixed_diffmerged"
    
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MuSiqQue_x4_UserControl_Final_merged"

    "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MuSiqQue_x4_UserControl_merged_version5_2_continueFinetune_checkpoint1090"
    )

for i in "${!SETTINGS[@]}"; do
    SETTING="${SETTINGS[$i]}"
    MERGE_PATH="${MERGE_PATHS[$i]}"

    Final_SETTING="${SETTINGS[$i]}_stage2"



    # DO THE COMPRESSION FOR LONGBENCH
    # conda activate vllm-multidoc
    conda activate multidoc_vllm
    # conda activate vllm-multidoc_2
    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench.py --name "${SETTING}" --model_path "${MERGE_PATH}" --parallel_size "${ALLOW_NUM_GPU}"


    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_UserControl_version5_Auto_stage1.py --name "${SETTING}" --model_path "${MERGE_PATH}" --parallel_size "${ALLOW_NUM_GPU}"

    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_UserControl_version5_Auto_stage2.py --name "${SETTING}" --model_path "${MERGE_PATH}" --parallel_size "${ALLOW_NUM_GPU}"



    # DO THE READING FOR LONGBENCH
    conda activate longbench
    # conda activate longbench_2


    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/pred.py --model "llama3.1-8b-instruct-128k" --e --setting "${Final_SETTING}"

    python /data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/eval.py --model "llama3.1-8b-instruct-128k_${Final_SETTING}"


done


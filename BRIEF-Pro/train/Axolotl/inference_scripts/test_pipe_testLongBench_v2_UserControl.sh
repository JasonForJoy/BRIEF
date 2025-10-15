
source /data2/junyizhang/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc


ALLOW_GPUS="4,5"
ALLOW_NUM_GPU="2"
SETTINGS=(
    "My_Axolotl_Finetuned_MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_checkpoint1047_5sentences_testAll"
    # "My_Axolotl_Finetuned_MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_checkpoint1047_Auto_testAll"
    )
MERGE_PATHS=(
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/lora-out-ins-epoch3-lr0_0001-hotpotOracle_10longer-merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/llama3_full_merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/llama3_full_merged_checkpoint-5624"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/MuSiqQue_x4_llama3_Final_full_merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/lora-out-ins-epoch3-lr0_0001-sent_wikifull_8longer_fixed_mixed_diffmerged"
    
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MuSiqQue_x4_UserControl_Final_merged"

    "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_merged_checkpoint1047"
    )

for i in "${!SETTINGS[@]}"; do
    SETTING="${SETTINGS[$i]}"
    MERGE_PATH="${MERGE_PATHS[$i]}"



    # DO THE COMPRESSION FOR LONGBENCH
    # conda activate vllm-multidoc
    conda activate multidoc_vllm
    # conda activate vllm-multidoc_2
    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench.py --name "${SETTING}" --model_path "${MERGE_PATH}" --parallel_size "${ALLOW_NUM_GPU}"


    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_testLongBench_v2_UserControl.py --name "${SETTING}" --model_path "${MERGE_PATH}" --parallel_size "${ALLOW_NUM_GPU}"





done


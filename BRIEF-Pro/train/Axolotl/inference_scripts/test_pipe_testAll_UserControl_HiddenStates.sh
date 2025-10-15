
source /data2/junyizhang/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc


ALLOW_GPUS="3"
ALLOW_NUM_GPU="1"
SETTINGS=(
    "Llama-3_2-3B-Instruct_noFinetune"
    # "My_finetune_MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_merged_checkpoint1047_20sentences_testAll"
    # "My_finetune_MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_merged_checkpoint1047_Auto_testAll"
    )
MERGE_PATHS=(
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/lora-out-ins-epoch3-lr0_0001-hotpotOracle_10longer-merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/llama3_full_merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/llama3_full_merged_checkpoint-5624"
    # "/data2/junyizhang/BRIEF_train_eval_Code/A_My_Code/MuSiqQue_x4_llama3_Final_full_merged"
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/lora-out-ins-epoch3-lr0_0001-sent_wikifull_8longer_fixed_mixed_diffmerged"
    
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MuSiqQue_x4_UserControl_Final_merged"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_merged_checkpoint1047"

    "meta-llama/Llama-3.2-3B-Instruct"
    )

for i in "${!SETTINGS[@]}"; do
    SETTING="${SETTINGS[$i]}"
    MERGE_PATH="${MERGE_PATHS[$i]}"



    # conda activate vllm-multidoc
    # conda activate multidoc_vllm
    # conda activate vllm-multidoc_2

    conda activate LLaVA_NeXT
    



    ###

    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_testAll_UserControl_HiddenStates.py --model_path "${MERGE_PATH}" --dataset_root "/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench" --out_dir "./HiddenStates_output/MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_merged_checkpoint1047_Auto_oridata/" --batch_size 1 --no_assistant_header

    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_testAll_UserControl_HiddenStates.py --model_path "${MERGE_PATH}" --dataset_root "/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench" --out_dir "./HiddenStates_output/MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_merged_checkpoint1047_Auto_ttt/" --batch_size 1 --no_assistant_header --datasets NQ_final_Test
    
    
    CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_testAll_UserControl_HiddenStates.py --model_path "${MERGE_PATH}" --dataset_root "/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench" --out_dir "./HiddenStates_output/Llama-3_2-3B-Instruct_noFinetune_oridata/" --batch_size 1 --no_assistant_header --datasets NQ_final_Test_10Doc


    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_testAll_UserControl_HiddenStates.py --model_path "${MERGE_PATH}" --dataset_root "/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench" --out_dir "./HiddenStates_output/MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_TaskType_checkpoint1047_Auto/" --batch_size 1 --no_assistant_header --datasets hotpotqa 2wikimqa musique

    # CUDA_VISIBLE_DEVICES="${ALLOW_GPUS}" python /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/inference_scripts/vllm_compression_longbench_testAll_UserControl_HiddenStates.py --model_path "${MERGE_PATH}" --dataset_root "/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench" --out_dir "./HiddenStates_output/MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_TaskType_checkpoint1047_Auto/" --batch_size 1 --no_assistant_header --datasets qasper narrativeqa multifieldqa_en


done


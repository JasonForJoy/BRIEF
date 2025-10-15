#!/bin/bash

# WANDB ENVIRONMENT VARIABLES
# Set the WANDB_DIR environment variable to specify where W&B stores generated files
# export WANDB_DIR="/local2/diwu/lyk/WandbFiles"
export WANDB_DIR="/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/WandbFiles"

# Set the WANDB_ARTIFACT_DIR environment variable to control where artifacts are stored
# export WANDB_ARTIFACT_DIR="./artifacts"
# export TMPDIR="/local2/diwu/lyk/WandbFiles/tmp"

export WANDB_ARTIFACT_DIR="/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/WandbFiles/artifacts"
export TMPDIR="/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/WandbFiles/tmp"


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
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-shorter.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-ori0.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-ori1.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-ori3.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-multihopRAG-ori5.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_o_in_i.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_o_in_i_noaneal.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0002-sent_cont_local_wiki_o_in_i_noaneal.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_pos_o_in_i.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0002-sent_cont_local_wiki_o_in_i.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0005-sent_cont_local_wiki_o_in_i.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_001-sent_cont_local_wiki_o_in_i.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0002-sent_cont_local_wiki_pos_o_in_i.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_pos_o_in_i_augment.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_o_in_i_augment.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_mixed.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_test_cir.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_test_cir_2.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_5_test_cir.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_mixed_diff.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-hotpotOracle_10longer.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_better_keep.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_better_half.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_5_better_keep.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_5_better_half.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-hotpotOracle_1longer_original.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-hotpotOracle_5longer.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_5_better_half.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-original-prune-hotpot-musique.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-original-prune-hotpot-musique_x4.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-hotpot-musique-orimix-inepoch.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-hotpot-musique-x4x10mix-inepoch.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-hotpotOracle_15longer.yml"
    # "/local2/diwu/lyk/Axolotl/examples/llama-3.2/series/datasets/factory/instruct-lora-3b-ep3-lr0_0001-hotpotOracle_20longer.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_ReproduceYuankai.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8.yml"
    
    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version2.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version3.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version4.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version5.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version5_Check1Loss.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version5_2.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version5_2_continueFinetune.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version6.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_version7.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_New_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-NQ_data_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-NQ_data.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-NQ_data_noCleanFuther.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-sent_cont_local_wiki_longer_8_singlehop.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-Qasper_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_Mix_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_Mix_controlWordDisMain10000_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_controlWordDisMain10000_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain20000_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_10kData_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain20000_10kData_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_Mix_normDistribution_controlDisMain10000_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_200FeqCut_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_Mix_normDistribution_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_200FeqCut_UserControl_continueFinetune.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_newPruneAnswer.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_UserControl_newPruneAnswer.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_SupportPassageAll.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_x6_Compare.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_x5_Compare.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_TaskType.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_Qasper_Mix_normDistribution_controlDisMain10000_UserControl_SourceType.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-Qasper_UserControl_continueFinetune_TaskType.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Qasper_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl_TaskType.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Qasper_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-LongAlign_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-Qasper_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Qasper_Mix_normDistribution_controlDisMain10000_UserControl_TaskType.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-LongAlign_Qasper_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_SampleBalance_UserControl_TaskType.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_MultiSingleMix.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_MultiSingleMix_TaskType.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_LLaMA_2_7B_chat_4k_drop.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_LLaMA_2_7B_chat_4k_truncate.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_LLaMA_2_7B_chat_4k_truncate_myTruncate.yml"

    # "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl_LLaMA_2_7B_chat_4k_truncate_myTruncate_drop4000.yml"

    "/data2/junyizhang/BRIEF_train_eval_Code/Axolotl/examples/llama-3.2/series/datasets/factory/My_instruct-lora-3b-ep3-lr0_0001-MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_controlDisMain10000_UserControl.yml"

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
        # CUDA_VISIBLE_DEVICES=0,6 accelerate launch --main_process_port 29523 -m axolotl.cli.train "$cfg" 2>&1 | while IFS= read -r line; do

        # CUDA_VISIBLE_DEVICES=3,4 accelerate launch -m axolotl.cli.train "$cfg" 2>&1 | while IFS= read -r line; do

        # CUDA_VISIBLE_DEVICES=1,6 accelerate launch --main_process_port 26345 -m axolotl.cli.train --resume-from-checkpoint /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MusiqueX8_HotpotX10_Mix_UserControl/checkpoint-788 "$cfg" 2>&1 | while IFS= read -r line; do

        CUDA_VISIBLE_DEVICES=1,2 accelerate launch --main_process_port 26345 -m axolotl.cli.train "$cfg" 2>&1 | while IFS= read -r line; do

        # CUDA_VISIBLE_DEVICES=3,4 accelerate launch -m axolotl.cli.train --resume-from-checkpoint /data2/junyizhang/BRIEF_train_eval_Code/Axolotl/outputs/My_finetune_MusiqueX8_HotpotX10_LongAlign_Mix_normDistribution_200FeqCut_UserControl/checkpoint-437 "$cfg" 2>&1 | while IFS= read -r line; do


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


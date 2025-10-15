


CUDA_VISIBLE_DEVICES=0,1 python /local2/diwu/lyk/Axolotl/inference_scripts/vllm_compression_longbench.py --name "HALF_EPOCH_MIX" --model_path "/local2/diwu/lyk/Axolotl/outputs/lora-out-ins-epoch3-lr0_0001-sent_wikifull_8longer_fixed_mixed_diff/merged"
"""
0714 This file is used to reply questions using vLLM, but it is not implemented with logit related feature. It can be used in Downstream performance check.

0715 We add the system prompt or job description such that it may perform better at TriviaQA tasks.
"""



import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if "vllm" not in conda_env:
    raise ValueError("Please double check whether this environment is vllm-environment! If your are confident, just comment in this line.")
from vllm import LLM, SamplingParams

import json


def arg_parse():
    parser = argparse.ArgumentParser(description="Inference a Llama3 8B using vLLM")
    parser.add_argument(
        "--parallel_size",
        type=int,
        default=2,
        help="Whether to use parallel."
    )
    parser.add_argument(
        "--instruct",
        action="store_false",
        help="Whether to use LLaMA 3 8B Instruct or the Base model."
    )
    args=parser.parse_args()
    return args
def main():
    args = arg_parse()
    with open("/home/jcgu/multidoc-reasoner-compressor/resources/MuSiQue/long_context/down_stream_performance/full/full_dev_document_str.json","r") as f:
        dataForInfer:list[dict] = json.load(f)

    inputstrs: list[str] = [temp['text'] for temp in dataForInfer]
    querys = [temp['question'] for temp in dataForInfer]


    if args.instruct == False:
        raise NotImplementedError("This part is not implemented yet.")
        llm = LLM(model="/home/jcgu/multidoc-reasoner-compressor/Llama-3.2-3B",tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9)
        tokenizer = llm.get_tokenizer()
    elif args.instruct == True:
        # INS_SUPPORT_PATH = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch3-2e-4-supporting-thirdchance/merged"
        # VANILLA_PATH = "/home/jcgu/multidoc-reasoner-compressor/Llama-3.2-3B-Instruct"
        # PROP_SUPPORT_PATH = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch3-2e-4-prop_supp/merged"
        # PRUNE_SUPPORT_PATH = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch3-2e-4-prune_supp-2/merged"

        ORACLE_EP1_1E_4 = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch1-lr0_0001-oracle_supporting/merged"
        ORACLE_EP2_1E_4 = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch2-lr0_0001-oracle_supporting/merged"
        ORACLE_EP2_2E_4 = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch2-lr0_0002-oracle_supporting/merged"
        ORACLE_EP2_5E_5 = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch2-lr5e-05-oracle_supporting/merged"
        ORACLE_EP3_2E_4 = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch3-lr0_0002-oracle_supporting/merged"
        EPS0_005_EP3_2E_4 = "/home/jcgu/multidoc-reasoner-compressor/resources/Axolotl/axolotl/outputs/lora-out-ins-epoch3-lr0_0002-prop_eps0_005_supporting/merged"

        llm = LLM(model=ORACLE_EP3_2E_4,tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9)
        tokenizer = llm.get_tokenizer()
    print("==================Model Tokeninzer Loaded==================")
    if args.instruct == True:
        prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': f"Write a high-quality summary of the provided documents with respect to the question. Question: {query}\n {inputstr} Summary:"}],tokenize=False,) for query,inputstr in zip(querys,inputstrs)]
        sampling_params = SamplingParams(temperature=0.01, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],max_tokens=512)
        outputs = llm.generate(prompts, sampling_params)
        replys = [output.outputs[0].text for output in outputs]
    else:
        prompts = [f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nWrite a high-quality summary of the provided documents with respect to the question.\n\n### Input:\nQuestion: {query}\n {inputstr} Summary:\n\n### Response:" for query,inputstr in zip(querys,inputstrs)]
        sampling_params = SamplingParams(temperature=0, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],max_tokens=512)
        outputs = llm.generate(prompts, sampling_params)
        replys = [output.outputs[0].text for output in outputs]

    for item,reply in zip(dataForInfer,replys):
        item['reply'] = reply
    

        
    # model_type = "vanilla"
    # model_type = "ins_support"
    # model_type = "prune_support"
    model_type = "ORACLE_EP3_2E_4"
    with open(f"/home/jcgu/multidoc-reasoner-compressor/resources/MuSiQue/long_context/down_stream_performance/full_dev_document_{model_type}_instruct_{args.instruct}.json","w") as f:
        dataForInfer = json.dump(dataForInfer,f,indent=4)



if __name__ == "__main__":
    main()
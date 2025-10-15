
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

def read_jsonl(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Strip any leading/trailing whitespace and parse the JSON object
                    data.append(json.loads(line.strip()))
            return data
def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
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
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="The model path."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="The model name."  
    )
    args=parser.parse_args()
    return args
def main():
    args = arg_parse()

    model_name = args.name
    model_path = args.model_path
    
    print(f"==================Model {model_name} Loaded==================")

    llm = LLM(model=model_path,tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9)
    tokenizer = llm.get_tokenizer()
    datasets = ["hotpotqa", "2wikimqa", "musique"]
    print("==================Model Tokeninzer Loaded==================")
    for dataset in datasets:
            
        dataForInfer:list[dict] = read_jsonl(f"/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench/unbiasdata/{dataset}.jsonl")

        inputstrs: list[str] = [temp['context'] for temp in dataForInfer]
        querys = [temp['input'] for temp in dataForInfer]


        if args.instruct == False:
            raise NotImplementedError("This part is not implemented yet.")

            
    
        if args.instruct == True:

            # control_length = 5


            prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': f"Write a high-quality summary of the provided documents with respect to the question.\n ### This is the question: {query}\n### These are the documents:\n{inputstr}\n### This is the summary:"}],tokenize=False,) for query,inputstr in zip(querys,inputstrs)]
            

            # control_prompt = f"Summarize the documents relevant to the question in K sentences, where K = <|reserved_special_token_100|>{control_length}<|reserved_special_token_101|>"

            control_prompt = f"Summarize the documents relevant to the question in K sentences, where K = <|reserved_special_token_100|>"

            prompts = [item + f"<|start_header_id|>assistant<|end_header_id|>\n\n{control_prompt}" for item in prompts]

            


            print("#################################################################################")
            print(prompts[0])
            print("#################################################################################")

            sampling_params = SamplingParams(temperature=0.01, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],max_tokens=1024)
            outputs = llm.generate(prompts, sampling_params)
            replys = [output.outputs[0].text for output in outputs]
        
        else:
            raise ValueError("Please specify the model type.")


        for item,reply in zip(dataForInfer,replys):
            reply = reply.removeprefix("<|start_header_id|>assistant<|end_header_id|>").strip()
            item['context'] = reply.removeprefix("<|start_header_id|>").strip().removeprefix("assistant").strip().removeprefix("<|end_header_id|>").strip().removeprefix(",").strip().removeprefix(".").strip()

        

        # model_type = "unbias_MULTI_RAG2"
        model_type = f"unbias_{model_name}"
        save_jsonl(dataForInfer,f"/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench/compdata/{dataset}_{model_type}.jsonl")
        # with open(f"/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench/compdata/{dataset}_{model_type}.json","w") as f:
        #     dataForInfer = json.dump(dataForInfer,f,indent=4)
        print(f"======================================================")



if __name__ == "__main__":
    main()
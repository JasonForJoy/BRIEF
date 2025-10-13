import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if "vllm" not in conda_env:
    raise ValueError("Please double check whether this environment is vllm-environment! If your are confident, just comment in this line.")
from vllm import LLM, SamplingParams
from statistics import mean
from prompt import llama_prompt,job_description,llama_fewshot_hotpot,llama_fewshot_nq,llama_fewshot_tqa,general_llama_prompt,llama_fewshot_musique,prolong_fewshot_musique
import json
import re

def arg_parse():
    parser = argparse.ArgumentParser(description="Inference a Llama3 8B using vLLM")
    parser.add_argument(
        "--proposition_name_or_path",
        type=str,
        default=None,
        help="Name or Path of propositions used for inferenced"
        )
    parser.add_argument(
        "--inference_type",
        type=str,
        default="ours",
        help="Specify which model to use. Choose from `ours`, `vanilla`, `all_passage` and `proposition`."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path to the json file. Example: `'0704_downstream_reply_proposition_logit_v3.json'`"
    )
    parser.add_argument(
        "--parallel_size",
        type=int,
        default=2,
        help="Whether to use parallel."
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        default="dir_name.txt",
        help="The default file that different files pass data."
    )
    parser.add_argument(
        "--downstream_dataset",
        type=str,
        default="tqa",
        choices=["tqa","nq","hotpot","musique"],
        help="tqa: TriviaQA, nq: Natural Questions, hotpot: Hotpot QA, musique: MuSiQue"
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether to use LLaMA 3 8B Instruct or the Base model."
    )
    args=parser.parse_args()
    dir_name = args.dir_name
    if args.proposition_name_or_path is None:
        print("[Warning]:Recieve no proposition input. Default to read from `dir_name.txt`.")
        with open(dir_name, "r") as f:
            dir_name = f.read().strip()
            args.proposition_name_or_path = os.path.join(dir_name,"reply.json")
    return args
def main():
    map_dict={"ours":["reply"], "vanilla":["reply"], "all_passage":["text"],"proposition":["summary"],"none":[]}
    args = arg_parse()

    with open(args.proposition_name_or_path,"r") as f:
        data = json.load(f)

    print(args.proposition_name_or_path)
    print(args.inference_type)
    data = [{"Idx":idx, "Question":d["question"],"QuestionId":d["question_id"],"Proposition":"" if args.inference_type == "none" else d[map_dict.get(args.inference_type)[0]]} for idx,d in enumerate(data)]

    if args.downstream_dataset == "tqa":
        fewshot_str = llama_fewshot_tqa
    elif args.downstream_dataset == "nq":
        fewshot_str = llama_fewshot_nq
    elif args.downstream_dataset == "hotpot":
        fewshot_str = llama_fewshot_hotpot
    elif args.downstream_dataset == "musique":
        fewshot_str = prolong_fewshot_musique

    
    prolong_prompt="""{few_shot}\n\n{document}\nQuestion: {question}\nAnswer:"""
    def getDocFromProp(prop:str):
        retDocString = ""
        propList = prop.split("\n")
        for item in propList:
            retDocString += f"Document: {item}\n"
        return retDocString

    prompts = [
        prolong_prompt.format(few_shot=fewshot_str,document=getDocFromProp(example['Proposition']),question=example['Question']) 
        if (example['Proposition'] != "" and example['Proposition'] != "This passage doesn't contain relevant information to the question.") 
        else prolong_prompt.format(few_shot=fewshot_str,document="",question=example['Question']) 
        for example in data
        ]


    if len(data) == 0:
        raise ValueError("Get empty input, please check your code!")

    if args.instruct == False:
        llm = LLM(model="shouldnotexist",tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9)
        tokenizer = llm.get_tokenizer()
    elif args.instruct == True:
        llm = LLM(model="ProLong-512k-Instruct",tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9)
        tokenizer = llm.get_tokenizer()
    print("==================Model Tokeninzer Loaded==================")
    # llm = LLM(model="Llama-3-8B",tensor_parallel_size=args.parallel_size)
    # tokenizer = llm.get_tokenizer()
    if args.instruct == True:
        # If use 8B-Instruct
        # See:https://github.com/meta-llama/llama3/blob/main/example_text_completion.py
        # The base model doesn't require prompt formatting
        # It is also normal, that the base model tends not to stop
        job_description = """Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\n"""
        prompts = [tokenizer.apply_chat_template([{'role': 'system', 'content': job_description},{'role': 'user', 'content': job_description+prompt}],tokenize=False,) for prompt in prompts]
    else:
        prompts = [job_description + prompt for prompt in prompts]
    print(prompts[0])
    # If use 8B-Instruct
    # prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],tokenize=False,) for prompt in prompts]
    # 271 is \n\n
    # See discussion at https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit
    # And at https://github.com/vllm-project/vllm/issues/4180
    # sampling_params = SamplingParams(temperature=0, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"),tokenizer.convert_tokens_to_ids("/n"),tokenizer.convert_tokens_to_ids("."),tokenizer.convert_tokens_to_ids(",")],max_tokens=32)
    sampling_params = SamplingParams(temperature=0, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],max_tokens=64)
    outputs = llm.generate(prompts, sampling_params)
    for d,output in zip(data,outputs):
        if args.instruct == False:
            match = re.search(r'[^.,\n]+', output.outputs[0].text)
            if match:
                d["Proposition_Reply"] = match.group(0)
            else:
                d["Proposition_Reply"] = ""
        elif args.instruct == True:
            output_text=output.outputs[0].text.removeprefix("<|start_header_id|>assistant:")
            match = re.search(r'[^.,\n]+', output_text)
            if match:
                d["Original_Reply"] = output.outputs[0].text.removeprefix("<|start_header_id|>assistant:")
                d["Proposition_Reply"] = match.group(0).strip().removeprefix("[").removesuffix("]")
            else:
                d["Original_Reply"] = output.outputs[0].text.removeprefix("<|start_header_id|>assistant:")
                d["Proposition_Reply"] = ""
            

        

    with open(args.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    main()
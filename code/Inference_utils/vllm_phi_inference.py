
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
from prompt import llama_prompt,job_description,llama_fewshot_hotpot,llama_fewshot_nq,llama_fewshot_tqa,general_llama_prompt
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
        default=1,
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
        choices=["tqa","nq","hotpot"],
        help="tqa: TriviaQA, nq: Natural Questions, hotpot: Hotpot QA"
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
    
    def get_final_prompt(fs,doc,ques):
        if not(doc != "" and doc != "This passage doesn't contain relevant information to the question.") :
            doc = "Irrelevant."
        if ques.strip() == doc.removeprefix("Passage: ").strip():
                    doc = "Irrelevant."
        l = []
        egs = fs.split("\n\n")
        for eg in egs:
            if eg != "":
                q = eg.split("\n")[0]
                a = eg.split("\n")[1]
                

                l.append({'role': 'user', 'content': job_description+q+"\nYour response should end with \"The answer is [the_answer]\" where the [the_answer] is the correct answer."})
                l.append({'role': 'assistant', 'content': f"The answer is {a.removeprefix('Answer: ')}"})
        l.append({'role': 'user', 'content': job_description+doc+"\n"+ques+"\nYour response should end with \"The answer is [the_answer]\" where the [the_answer] is the correct answer."})
        l.append({'role': 'assistant', 'content': f"The answer is"})
        return l

    # prompts = [
    #     general_llama_prompt.format(few_shot=fewshot_str,document=f"Passage: {example['Proposition']}",question=example['Question']) 
    #     if (example['Proposition'] != "" and example['Proposition'] != "This passage doesn't contain relevant information to the question.") 
    #     else general_llama_prompt.format(few_shot=fewshot_str,document="",question=example['Question']) 
    #     for example in data
    #     ]
    prompts = [
        get_final_prompt(fs=fewshot_str,doc=f"Passage: {example['Proposition']}",ques=example["Question"]) 
        for example in data
        ]

        


    if len(data) == 0:
        raise ValueError("Get empty input, please check your code!")

    phi_model = "phi3-mini"
    # phi_model = "phi3-small"
    # phi_model = "phi3-medium"

    if args.instruct == False:
        llm = LLM(model="hahaha",tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9,trust_remote_code=True)
        tokenizer = llm.get_tokenizer()
    elif args.instruct == True:
        llm = LLM(model=phi_model,tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9,trust_remote_code=True)
        tokenizer = llm.get_tokenizer()
    print("==================Model Tokeninzer Loaded==================")
    # llm = LLM(model="Llama-3-8B",tensor_parallel_size=args.parallel_size)
    # tokenizer = llm.get_tokenizer()
    if args.instruct == True:
        # If use 8B-Instruct
        # See:https://github.com/meta-llama/llama3/blob/main/example_text_completion.py
        # The base model doesn't require prompt formatting
        # It is also normal, that the base model tends not to stop
        new_prompts = []
        for p in prompts:
            try:
                tk = tokenizer.apply_chat_template(p,tokenize=False,) 
                new_prompts.append(tk)
            except Exception as e:
                print(e)
                # print(p)

        prompts = [p.removesuffix("<|endoftext|>").strip().removesuffix("<|end|>") for p in new_prompts]

        # prompts = [tokenizer.apply_chat_template(prompt,tokenize=False,) for prompt in prompts]
        
        # prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': job_description+prompt}],tokenize=False,) for prompt in prompts]
    # else:
        # prompts = [job_description + prompt for prompt in prompts]
    # If use 8B-Instruct
    # prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],tokenize=False,) for prompt in prompts]
    # 271 is \n\n
    # See discussion at https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit
    # And at https://github.com/vllm-project/vllm/issues/4180
    sampling_params = SamplingParams(temperature=0, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")],max_tokens=64)

    outputs = llm.generate(prompts, sampling_params)
    for d,output,pp in zip(data,outputs,prompts):
        if args.instruct == False:
            match = re.search(r'[^.,\n]+', output.outputs[0].text)
            if match:
                d["Proposition_Reply"] = match.group(0)
            else:
                d["Proposition_Reply"] = ""
        elif args.instruct == True:
            output_text=output.outputs[0].text.removeprefix("<|assistant|>\n\n")
            d["Original_Prompt"] = pp
            d["Original_Output"] = output_text
            
            if "i apologize for the mistake" in output_text.lower() or "i am sorry" in output_text.lower():
                if "the answer is" in output_text.lower():
                    d["Proposition_Reply"] = output_text.lower().split("the answer is")[1].strip()
                else:
                    d["Proposition_Reply"] = ""
            else:
                d["Proposition_Reply"] = output_text.lower().removeprefix("the answer is").strip()
            

        

    with open(args.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    main()
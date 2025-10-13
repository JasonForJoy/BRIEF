"""
This file can be used to do inference as well as get the related logits.
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from vllm import LLM, SamplingParams
import argparse
import re
from tqdm import tqdm
from statistics import mean
from custom_utils import normalize_answer
from prompt import prompt,llama_prompt,job_description
import json




job_desc = "Answer the question:\n"
fs = """Question: Which British politician was the first person to be made an Honorary Citizen of the United States of America?\nAnswer: Winston Churchill\n\nQuestion: Which event of 1962 is the subject of the 2000 film Thirteen Days starring Kevin Costner?\nAnswer: The Cuban Missile Crisis\n\nQuestion: Which country hosted the 1968 Summer Olympics?\nAnswer: Mexico\n\nQuestion: In which city did the assassination of Martin Luther King?\nAnswer: MEMPHIS, Tennessee\n\nQuestion: Which German rye bread is named, according to many reliable sources, from the original meaning 'Devil's fart'?\nAnswer: Pumpernickel"""

def arg_parse():
    parser = argparse.ArgumentParser(description="Inference a Llama using vllm")
    parser.add_argument(
        "--proposition_name_or_path",
        type=str,
        default=None,
        help="Name or Path of propositions used for inferenced"
        )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path to the json file."
    )
    parser.add_argument(
        "--parallel_size",
        type=int,
        default=4,
        help="Whether to use parallel."
    )
    parser.add_argument(
        "--test",
        action='store_true',
        help="Whether set test mode"
    )
    parser.add_argument(
        "--instruct",
        action='store_true',
        help="Whether to use LLaMA-3 8B Instruct"
    )
    args=parser.parse_args()

    return args
def main():
    args = arg_parse()
    print(f"Type is {isinstance(args.instruct,bool)} bool,val is {args.instruct}")
    with open(args.proposition_name_or_path,"r") as f:
        data = json.load(f)

    if args.test is True:
        print("test")
        data = data[:100]

    data = [{"Question":d["Question"],"QuestionId":d["QuestionId"],"Proposition":d["Proposition"]} for d in data]
    
    if len(data) == 0:
        raise ValueError("Get empty input, please check your code!")
    if args.instruct == False:
        prompts = [llama_prompt.format(document=f"Passage: {example['Proposition']}",question=example['Question']) if example['Proposition'] != "" else llama_prompt.format(document=example['Proposition'],question=example['Question']) for example in data]
    else:

        def get_final_prompt(doc,ques):
            l = []
            egs = fs.split("\n\n")
            for eg in egs:
                if eg != "":
                    q = eg.split("\n")[0]
                    a = eg.split("\n")[1]
                    

                    l.append({'role': 'user', 'content': job_desc+q+"\nYour response should end with \"The answer is [the_answer]\" where the [the_answer] is the correct answer."})
                    l.append({'role': 'assistant', 'content': f"The answer is {a.removeprefix('Answer: ')}"})
            if doc.strip() == "Passage:":
                doc = ""
            
            l.append({'role': 'user', 'content': job_desc+doc+"\n"+ques+"\nYour response should end with \"The answer is [the_answer]\" where the [the_answer] is the correct answer."})
            l.append({'role': 'assistant', 'content': f"The answer is"})
            return l

        
        prompts = [
            get_final_prompt(
                doc=f"Passage: {example['Proposition']}",
                ques=example["Question"])
            for example in data
        ]
    print("==================Prompt Prepared==================")

    # Default gpu_utilization 0.9
    if args.instruct == False:
        llm = LLM(model="Llama-3-8B",tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9)
        tokenizer = llm.get_tokenizer()
        print("==================Model Tokeninzer Loaded==================")
    elif args.instruct == True:
        llm = LLM(model="Llama-3.1-8B-Instruct",tensor_parallel_size=args.parallel_size,gpu_memory_utilization=0.9)
        tokenizer = llm.get_tokenizer()
        print("==================Model Tokeninzer Loaded (Instruct)==================")

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
        prompts = new_prompts
        # prompts = [tokenizer.apply_chat_template([{'role': 'system', 'content': job_description.strip().removesuffix(":") + " Directly give the answer with no prefix or suffix:"},{'role': 'user', 'content': prompt}],tokenize=False,) for prompt in prompts]
        print("==================Prompt Loaded (Instruct)==================")
    else:
        # raise NotImplementedError
        prompts = [job_description + prompt for prompt in prompts]
        print("==================Prompt Loaded ==================")


    # 271 is \n\n
    # See discussion at https://huggingface.co/astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit
    # And at https://github.com/vllm-project/vllm/issues/4180
    if args.instruct == True:
        print("==================Generating Outputs ==================")
        sampling_params = SamplingParams(temperature=0, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],max_tokens=64,logprobs=0)
        
        outputs = llm.generate(prompts, sampling_params)
        print("==================Generated Outputs, Post Processing ...==================")
        for d,output,pp in zip(tqdm(data),outputs,prompts):
            logprobs = output.outputs[0].logprobs
            decoded_tokens = [next(iter(p.values())).decoded_token for p in logprobs]
            # llama-3-8b instruct takes 4 tokens
            # but it seems to only take 3 tokens with llama-3.1-8b
            temp = output.outputs[0].text
            if temp.startswith("<|start_header_id|>assistant<|end_header_id|>"):
                header = 4
            else:
                header = 3

            decoded_tokens = decoded_tokens[header:]
            logprobs = logprobs[header:]
            logprobs = [next(iter(p.values())).logprob for p in logprobs]
            # print(decoded_tokens)

            # The assistant prefix is 4 tokens, so we start from 4, we use it in the legacy version to get the answer!

            # newline_position = next((i for i, token in enumerate(decoded_tokens) if ("\n" in token) or("." == token) or ("," == token) or ("" == token)), len(decoded_tokens))
            # # print(newline_position)
            # filtered_logprobs = [next(iter(p.values())).logprob for p in logprobs[:newline_position]]
            
            output_text=output.outputs[0].text.removeprefix("<|start_header_id|>assistant<|end_header_id|>\n\n").removeprefix("<|start_header_id|>assistant\n\n")

            # d["Original_Prompt"] = pp
            d["Original_Output"] = output_text


            def get_list_loc_of_str(l:list,s:str):
                l = [item.lower() for item in l ]
                start_loc = 0
                end_loc = len(l)
                current_l = "".join(l[start_loc:end_loc])
                try:
                    # sanity check
                    assert s != ""
                    assert s in current_l
                except Exception as e:
                    print(f"error is:{e}")
                    print(f"list is:{l}")
                    print(f"str is {s}")
                    return None,None
            

                while s in current_l.lower():
                    start_loc += 1
                    current_l = "".join(l[start_loc:end_loc])
                start_loc -= 1
                current_l = "".join(l[start_loc:end_loc])
                while s in current_l.lower():
                    end_loc -= 1
                    current_l = "".join(l[start_loc:end_loc])
                end_loc += 1
                current_l = "".join(l[start_loc:end_loc])
                return start_loc, end_loc



            if "i apologize for the mistake" in output_text.lower() or "i am sorry" in output_text.lower():
                if "the answer is" in output_text.lower():
                    save_ans = output_text.lower().split("the answer is")[1].strip()
                    if save_ans != "":
                        d["Proposition_Reply"] = save_ans
                        start,end = get_list_loc_of_str(l=decoded_tokens,s=save_ans)
                        if start == None and end == None:
                            d["Log_Probs"]=[]
                            d["Unnormalize_Seq_Probs"] = None
                            d["Len_Seq_Probs"] = None
                        else:
                            log_prob = logprobs[start:end]
                            
                            # d["Decoded_Ans"] = decoded_tokens[start:end]
                            d["Log_Probs"]=log_prob
                            d["Unnormalize_Seq_Probs"] = sum(log_prob)
                            d["Len_Seq_Probs"] = mean(log_prob)
                    else:
                        d["Proposition_Reply"] = ""
                        # d["Decoded_Ans"] = []
                        d["Log_Probs"]=[]
                        d["Unnormalize_Seq_Probs"] = None
                        d["Len_Seq_Probs"] = None
                else:
                    d["Proposition_Reply"] = ""
                    # d["Decoded_Ans"] = []
                    d["Log_Probs"]=[]
                    d["Unnormalize_Seq_Probs"] = None
                    d["Len_Seq_Probs"] = None
            else:
                if output_text.lower().startswith("the answer is"):                
                    save_ans = output_text.lower().removeprefix("the answer is").strip()
                elif "the answer is" in output_text.lower():
                    save_ans = output_text.lower().split("the answer is")[1].strip()
                else:
                    save_ans = output_text.lower().strip()

                if save_ans != "":

                    d["Proposition_Reply"] = save_ans
                    start,end = get_list_loc_of_str(l=decoded_tokens,s=save_ans)
                    if start == None and end == None:
                        d["Log_Probs"]=[]
                        d["Unnormalize_Seq_Probs"] = None
                        d["Len_Seq_Probs"] = None
                    else:
                        log_prob = logprobs[start:end]
                        # d["Decoded_Ans"] = decoded_tokens[start:end]
                        d["Log_Probs"]=log_prob
                        d["Unnormalize_Seq_Probs"] = sum(log_prob)
                        d["Len_Seq_Probs"] = mean(log_prob)
                else:
                    d["Proposition_Reply"] = ""
                    # d["Decoded_Ans"] = []
                    d["Log_Probs"]=[]
                    d["Unnormalize_Seq_Probs"] = None
                    d["Len_Seq_Probs"] = None

    
    elif args.instruct == False:
        sampling_params = SamplingParams(temperature=0, top_p=0.95,stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"),tokenizer.convert_tokens_to_ids("\n"),tokenizer.convert_tokens_to_ids(","),tokenizer.convert_tokens_to_ids(".")],max_tokens=32,logprobs=0)
        outputs = llm.generate(prompts, sampling_params)
        
        for d,output in zip(data,outputs):
            logprobs = output.outputs[0].logprobs
            decoded_tokens = [next(iter(p.values())).decoded_token for p in logprobs]
            newline_position = next((i for i, token in enumerate(decoded_tokens) if ("\n" in token) or("." == token) or ("," == token) or ("" == token)), len(decoded_tokens))
            filtered_logprobs = [next(iter(p.values())).logprob for p in logprobs[:newline_position]]
            if filtered_logprobs == [] or filtered_logprobs == None:
                d["Proposition_Reply"] = ""
                d["Log_Probs"]=[]
                d["Unnormalize_Seq_Probs"] = None
                d["Len_Seq_Probs"] = None
            else:
                match = re.search(r'[^.,\n]+', output.outputs[0].text)
                if match:
                    d["Proposition_Reply"] = match.group(0)
                    d["Log_Probs"]=filtered_logprobs
                    d["Unnormalize_Seq_Probs"]=sum(d["Log_Probs"])
                    d["Len_Seq_Probs"]=mean(d["Log_Probs"])
                else:
                    d["Proposition_Reply"] = ""
                    d["Log_Probs"]=[]
                    d["Unnormalize_Seq_Probs"] = None
                    d["Len_Seq_Probs"] = None
        
            
        
    print("==================Finished, Saving ==================")
    print(f"=================={args.output_path} ==================")
    with open(args.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    main()

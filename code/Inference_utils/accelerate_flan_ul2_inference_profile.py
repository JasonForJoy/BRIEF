import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from accelerate import Accelerator,InitProcessGroupKwargs
from accelerate import ProfileKwargs
# ProfileKwargs is for profiling, and we recommend using this under the latest version of accelerate
from accelerate.utils import gather_object
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from prompt import prompt,general_prompt,flan_fewshot_tqa,flan_fewshot_hotpot,flan_fewshot_nq,flan_fewshot_musique
from transformers import T5Tokenizer, T5ForConditionalGeneration
from statistics import mean
from Utilities.json_utils import read_jsonl
import torch, time, json
import argparse 
import json

# with open("../data/proposition_batch.json","r") as f:
#     data = json.load(f)
#################################################
#################################################
# For downstream performance

def arg_parse():
    parser = argparse.ArgumentParser(description="Inference a Flan UL2 using Accelerate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per device batch size"
        )
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
        "--downstream_dataset",
        type=str,
        default="tqa",
        choices=["tqa","nq","hotpot","musique"],
        help="tqa: TriviaQA, nq: Natural Questions, hotpot: Hotpot QA, musique: MuSiQue"
    )
    args=parser.parse_args()
    if args.proposition_name_or_path is None:
        with open("dir_name.txt", "r") as f:
            dir_name = f.read().strip()
            args.proposition_name_or_path = os.path.join(dir_name,"reply.json")
    return args

def main():
    map_dict={"ours":["reply"], "vanilla":["reply"], "all_passage":["text"],"proposition":["summary"],"none":[]}
    args = arg_parse()
    print(f"Save at: ../FLOPS/{args.downstream_dataset}_{args.outpath.removesuffix('.json').replace('/','_')}_readtop5_flop.log")
    BATCH_SIZE = args.batch_size
    try:
        with open(args.proposition_name_or_path,"r") as f:
            data = json.load(f)
    except Exception as e:
        print(e)
        data = read_jsonl(args.proposition_name_or_path)

    if args.inference_type != "none":
        data = [{"Idx":idx, "Question":d["question"],"QuestionId":d["question_id"],"Proposition":d[map_dict.get(args.inference_type)[0]]} for idx,d in enumerate(data)]
    else:
        data = [{"Idx":idx, "Question":d["question"],"QuestionId":d["question_id"],"Proposition":""} for idx,d in enumerate(data)]


    if len(data) == 0:
        raise ValueError("Get empty input, please check your code!")
    # Here you can adjust the timeout to prevent unexpected termination of your code.

    # kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    # For profiling
    kwargs = [ProfileKwargs(with_flops=True),InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    # model_name = "google-t5/t5-small"
    model_name = "google/flan-ul2"

    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map={"": accelerator.process_index})
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # batch, left pad (for inference), and tokenize
    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]  
        for prompt_batch in batches:
            if args.downstream_dataset == "tqa":
                few_shot_string=flan_fewshot_tqa
            elif args.downstream_dataset == "hotpot":
                few_shot_string=flan_fewshot_hotpot
            elif args.downstream_dataset == "nq":
                few_shot_string=flan_fewshot_nq
            elif args.downstream_dataset == "musique":
                few_shot_string=flan_fewshot_musique
            else:
                raise ValueError("Bug here, unknown downstream dataset")
            p_prompt_batch = [
                    general_prompt.format(few_shot=few_shot_string,document=example['Proposition'],question=example['Question']) 
                    if (example['Proposition'] != "" and example['Proposition'] != "This passage doesn't contain relevant information to the question.")
                    else prompt.format(few_shot=few_shot_string,document="",question=example['Question'])
                    for example in prompt_batch
                    ]
            tokenized_inputs = tokenizer(p_prompt_batch, padding=True, truncation=True,return_tensors="pt")
            tokenized_inputs["Idx"] = [example["Idx"] for example in prompt_batch]
            batches_tok.append(tokenized_inputs)
        return batches_tok

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    
    save_p = None
    with accelerator.profile() as prof: 
    
        # divide the prompt list onto the available GPUs 
        with accelerator.split_between_processes(data) as prompts:
            results=[]

            # have each GPU do inference in batches
            prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=BATCH_SIZE)

            for prompts_tokenized in tqdm(prompt_batches):
                ids = prompts_tokenized['Idx']
                input_ids = prompts_tokenized['input_ids'].to(accelerator.device)
                attention_mask = prompts_tokenized['attention_mask'].to(accelerator.device)
                # with accelerator.profile() as prof: 
                outputs_tokenized=model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512)

                

                # count and decode gen. tokens 
                outputs=tokenizer.batch_decode(outputs_tokenized,skip_special_tokens=True)
                for i, output_text in enumerate(outputs):
                    original_data = next(item for item in prompts if item["Idx"] == ids[i])
                    original_data["Proposition_Reply"]=output_text
                    results.append(original_data)


        save_p = prof

    # Prevent timeout
    accelerator.wait_for_everyone()  
    # collect results from all the GPUs
    results_gathered=gather_object(results)
    

    if accelerator.is_main_process:
        with open(args.output_path, 'w') as json_file:
            json.dump(results_gathered, json_file, indent=4)

        print("Profiling flops!")
        with open(f"../FLOPS/{args.downstream_dataset}_{args.outpath.removesuffix('.json').replace('/','_')}_readtop5_flop.log","w") as f:
            f.write(save_p.key_averages().table(sort_by="flops", row_limit=50))



if __name__ == "__main__":
    main()

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from accelerate import Accelerator,InitProcessGroupKwargs
# from accelerate import ProfileKwargs #####
from accelerate.utils import gather_object
from datetime import timedelta
# from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from prompt import prompt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from statistics import mean
# import torch, time, json
import argparse
from datetime import datetime
import json

# To adapt this script, just change the `prepare_prompt` function. Be sure to deal with padding and truncation carefully.

def parse_args():
    parser = argparse.ArgumentParser(description="Inference a pretrained T5 model using Accelerate")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The name of T5 models to use or the path to the pretrained model. Example: google/flan-ul2, google/flan-t5-large"
        )
    parser.add_argument(
        "--use_cache",
        type=str,
        default=None,
        help=("If use cache dir")
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name of tokenizers the T5 model is to use. If not passed, defaults to `model_name`"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The path to the dataset to be tested/evaluated. Currently only supports `json` format of a list of dicts. Each dict should have field"
    )
    parser.add_argument(
        "--timeout_limit",
        type=int,
        default=600,
        help="The timeout limit for accelerate communication."
    )
    parser.add_argument(
        "--per_device_batchsize",
        type=int,
        default=1,
        help="The batchsize used per device."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="The maximum number of tokens to be generated."
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default=None,
        help="The path to save the inference result"
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="",
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="Max input sequence length."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams used when decoding"
    )
    parser.add_argument(
        "--dir_name",
        type=str,
        default="dir_name.txt",
        help="Intermediate placec to pass variable between files"
    )

    args = parser.parse_args()
    args.tokenizer_name = args.model_name if args.tokenizer_name is None else args.tokenizer_name
    if args.model_name is None:
        raise ValueError("Need to specific either a huggingface model or a path to local model.")
    if args.dataset_path is None:
        raise ValueError("Need either a dataset name for a test/validation file.")
    if args.output_file_path is None:
        raise ValueError("Need to spefic an output path. Will create if not already exist.")
    args.output_file_path = '{}-{}-{}'.format(
        args.output_file_path, 
        args.model_name.replace("/", "-"),
        datetime.now().strftime("%Y-%m-%d_%H-%M")
        )
    args.output_file_path
    dir_name = args.dir_name
    with open(dir_name, "w") as f:
        f.write(args.output_file_path)
    return args



def main():
    try:
        print(os.environ["CUDA_VISIBLE_DEVICES"])
    except Exception as e:
        print(f"{e} happens!")
    args = parse_args()
    # Here you can adjust the timeout to prevent unexpected termination of your code.
    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout_limit))]
    # kwargs = [ProfileKwargs(with_flops=True),InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    accelerator = Accelerator(kwargs_handlers=kwargs)
    if args.use_cache is None:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map={"": accelerator.process_index})
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map={"": accelerator.process_index},cache_dir=args.use_cache)
        tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name,cache_dir=args.use_cache)
        

    # with open(args.dataset_path,"r") as f:
    #     data = json.load(f)
    # print(args.tokenizer_name)
    
    def read_jsonl(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip any leading/trailing whitespace and parse the JSON object
                data.append(json.loads(line.strip()))
        return data
    data = read_jsonl(args.dataset_path)

    data = [{"Idx":idx, "question":d["question"],"text":d["text"],"question_id":d["question_id"],"summary":d["summary"]}for idx,d in enumerate(data)]


    # batch, left pad (for inference), and tokenize
    def prepare_prompts(prompts, tokenizer, batch_size=16,flan_prefix=""):
        if "google-t5/t5-large/notpossible" in args.model_name:
            print("You are using the vanilla Flan-T5 large. We try only using the `summarize: ` prefix when summarizing without the query awareness.")
            batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
            batches_tok=[]  
            for prompt_batch in batches:
                inputs = [example['text'] for example in prompt_batch]
                p_prompt_batch = [flan_prefix + inp for inp in inputs]
                tokenized_inputs = tokenizer(p_prompt_batch,max_length=args.max_source_length, padding='max_length',truncation=True,return_tensors="pt")
                # tokenized_inputs = tokenizer(p_prompt_batch, padding=True, truncation=True,return_tensors="pt")
                tokenized_inputs["Idx"] = [example["Idx"] for example in prompt_batch]
                batches_tok.append(tokenized_inputs)
        else:
            batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
            batches_tok=[]  
            for prompt_batch in batches:
                inputs = ["Question: {}\n Document: {}\n Summary: ".format(example['question'],example['text']) for example in prompt_batch]
                p_prompt_batch = [flan_prefix + inp for inp in inputs]
                tokenized_inputs = tokenizer(p_prompt_batch,max_length=args.max_source_length, padding='max_length',truncation=True,return_tensors="pt")
                # tokenized_inputs = tokenizer(p_prompt_batch, padding=True, truncation=True,return_tensors="pt")
                tokenized_inputs["Idx"] = [example["Idx"] for example in prompt_batch]
                batches_tok.append(tokenized_inputs)
        return batches_tok

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    

    # divide the prompt list onto the available GPUs 

    # save_p = None
    # with accelerator.profile() as prof:


    with accelerator.split_between_processes(data) as prompts:
        results=[]

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=args.per_device_batchsize,flan_prefix=args.source_prefix)

        for prompts_tokenized in tqdm(prompt_batches):
            ids = prompts_tokenized['Idx']
            input_ids = prompts_tokenized['input_ids'].to(accelerator.device)
            attention_mask = prompts_tokenized['attention_mask'].to(accelerator.device)
            
            outputs_tokenized=model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens)


            # count and decode gen. tokens 
            outputs=tokenizer.batch_decode(outputs_tokenized,skip_special_tokens=True)
            for i, output_text in enumerate(outputs):
                original_data = next(item for item in prompts if item["Idx"] == ids[i])
                original_data["reply"]=output_text
                results.append(original_data)
    

    # Prevent timeout
    accelerator.wait_for_everyone()  
    # collect results from all the GPUs
    results_gathered=gather_object(results)
        # save_p = prof

    if accelerator.is_main_process:
        os.makedirs(args.output_file_path, exist_ok=True)
        with open(os.path.join(args.output_file_path,"reply.json"), 'w') as json_file:
            json.dump(results_gathered, json_file, indent=4)

        # print("Profiling flops!")
        # with open("../FLOPS/summary_flop_batch_1.log","w") as f:
        #     f.write(save_p.key_averages().table(sort_by="flops", row_limit=50))

if __name__ == "__main__":
    main()

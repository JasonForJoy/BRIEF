import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from accelerate import Accelerator,InitProcessGroupKwargs
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from Utilities.json_utils import json_fix_quotes
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from statistics import mean
import torch, time, json
from datetime import timedelta
import json
import logging
import argparse
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("pipeline.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
def arg_parse():
    parser = argparse.ArgumentParser(description="Inference a Flan UL2 using Accelerate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size"
        )
    parser.add_argument(
        "--inpath",
        type=str,
        default=None,
        help="Input path to the json file. Must be a list of dicts."
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Output path to the json file. Will be a list of dicts."
    )
    args=parser.parse_args()
    return args

def main():
    args = arg_parse()
    BATCH_SIZE = args.batch_size
    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    model_name = "chentong00/propositionizer-wiki-flan-t5-large"
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,    
        device_map={"": accelerator.process_index},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)   

    with open(args.inpath,"r") as f:
        data = json.load(f)

    # batch, left pad (for inference), and tokenize
    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]  
        for prompt_batch in batches:
            p_prompt_batch = [f"Title: {example['Title']}. Section: . Content: {example['Proposition']}" for example in prompt_batch]
            tokenized_inputs = tokenizer(p_prompt_batch, max_length=256, padding='max_length',truncation=True, return_tensors="pt")
            tokenized_inputs["Idx"] = [example["Idx"] for example in prompt_batch]
            batches_tok.append(tokenized_inputs)
        return batches_tok

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()    

    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(data) as prompts:
        results=[]

        # have each GPU do inference in batches
        prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=BATCH_SIZE)

        for prompts_tokenized in tqdm(prompt_batches):
            ids = prompts_tokenized['Idx']
            input_ids = prompts_tokenized['input_ids'].to(accelerator.device)
            attention_mask = prompts_tokenized['attention_mask'].to(accelerator.device)

            outputs_tokenized=model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512,num_beams=3,early_stopping=True)

            # remove prompt from gen. tokens, Deprecated
            # outputs_tokenized=[ tok_out[len(tok_in):] 
            #     for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            outputs=tokenizer.batch_decode(outputs_tokenized,skip_special_tokens=True)
            for i, output_text in enumerate(outputs):
                original_data = next(item for item in data if item["Idx"] == ids[i])
                try:
                    prop_list = json.loads(output_text)
                except json.JSONDecodeError:
                    if (d:=json_fix_quotes(output_text)):
                        prop_list = d
                    elif (d:=json_fix_quotes(original_data["Proposition"])):
                        prop_list = d
                    else:
                        prop_list = [f"FAIL_PARSE: {output_text}"]
                        logger.warning(f"Failed to parse output text as JSON for ID {ids[i]}. The text is {output_text[:15]} ...")
                if len(prop_list) == 0:
                    prop_list = [f"FAIL_PARSE: {output_text}"]
                    logger.warning(f"Parse [] for ID {ids[i]}. The text is {output_text[:15]} ...")
                
                original_data["Propositions"]=prop_list
                results.append(original_data)


    # collect results from all the GPUs
    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        with open(args.outpath, 'w') as json_file:
            json.dump(results_gathered, json_file, indent=4)

if __name__ == "__main__":
    main()
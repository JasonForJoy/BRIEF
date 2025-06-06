import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from accelerate import Accelerator,InitProcessGroupKwargs
from accelerate.utils import gather_object
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from prompt import prompt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from statistics import mean
import torch, time, json
import json
import argparse
import math


def arg_parse():
    parser = argparse.ArgumentParser(description="Inference a Flan UL2 using Accelerate")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per device batch size"
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
    with open(args.inpath,"r") as f:
        data = json.load(f)

    # Sanity check
    assert 'Question' in data[0].keys()
    assert 'Proposition' in data[0].keys()
    assert 'QuestionId' in data[0].keys()
    data= [{"Idx":idx,**d}for idx,d in enumerate(data)]
    
    if len(data) == 0:
        raise ValueError("Get empty input, please check your code!")
    # Here you can adjust the timeout to prevent unexpected termination of your code.
    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    model_name = "google/flan-ul2"

    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map={"": accelerator.process_index})
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # batch, left pad (for inference), and tokenize
    def prepare_prompts(prompts, tokenizer, batch_size=16):
        batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok=[]  
        for prompt_batch in batches:
            p_prompt_batch = [prompt.format(document=example['Proposition'],question=example['Question']) for example in prompt_batch]
            tokenized_inputs = tokenizer(p_prompt_batch, padding=True, truncation=True,return_tensors="pt")
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

            outputs_tokenized=model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=32,return_dict_in_generate=True, output_scores=True)

            sequences = outputs_tokenized.sequences
            sequence_scores = outputs_tokenized.scores

            # count and decode gen. tokens 
            outputs=tokenizer.batch_decode(sequences,skip_special_tokens=True)
            for i, output_text in enumerate(outputs):
                original_data = next(item for item in prompts if item["Idx"] == ids[i])
                original_data["Proposition_Reply"]=output_text
                # Compute token probabilities
                token_probs = []
                sequence = sequences[i]
                generated_tokens = tokenizer.decode(sequence, skip_special_tokens=True).split()
                for j in range(len(generated_tokens)):
                    token_logits = sequence_scores[j][i]
                    token_id = tokenizer.encode(generated_tokens[j], add_special_tokens=False)[0]
                    token_prob = torch.softmax(token_logits, dim=-1)[token_id].item()
                    token_probs.append(token_prob)
                token_probs = torch.log(torch.tensor(token_probs)).tolist()
                    
                # Calculate sequence probability as the product of token probabilities
                if len(token_probs) >= 1:
                    sequence_probability = mean(token_probs)  # Using mean to avoid extremely small values
                else:
                    sequence_probability = None
                original_data["Log_Probabilities"] = token_probs
                original_data["Sequence_Probability"] = sequence_probability
                results.append(original_data)

    # Prevent timeout
    accelerator.wait_for_everyone()  
    # collect results from all the GPUs
    results_gathered=gather_object(results)

    if accelerator.is_main_process:
        with open(args.outpath, 'w') as json_file:
            json.dump(results_gathered, json_file, indent=4)

if __name__ == "__main__":
    main()
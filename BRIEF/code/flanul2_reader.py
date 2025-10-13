#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from accelerate import Accelerator,InitProcessGroupKwargs
# from accelerate import ProfileKwargs
# ProfileKwargs is for profiling, and we recommend using this under the latest version of accelerate
from accelerate.utils import gather_object
from datetime import timedelta
from tqdm import tqdm
from prompt import general_prompt,flan_fewshot_tqa,flan_fewshot_hotpot,flan_fewshot_nq,flan_fewshot_musique
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Utilities.json_utils import read_jsonl
import json
import argparse 

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
        help="Specify which model to use. Choose from `ours`, `vanilla`, `all_passage`, `proposition` and `none`."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path to the json file."
    )
    parser.add_argument(
        "--downstream_dataset",
        type=str,
        default="tqa",
        choices=["tqa","nq","hotpot","musique","multihop-nq","multihop-tqa"],
        help="tqa: TriviaQA, nq: Natural Questions, hotpot: Hotpot QA, musique: MuSiQue, multihop-nq: Multihop-NQ, multihop-tqa: Multihop-TriviaQA"
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

    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]

    # For profiling
    # kwargs = [ProfileKwargs(with_flops=True),InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

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
            elif args.downstream_dataset in ["hotpot", "multihop-nq", "multihop-tqa"]:
                few_shot_string=flan_fewshot_hotpot
            elif args.downstream_dataset == "nq":
                few_shot_string=flan_fewshot_nq
            elif args.downstream_dataset == "musique":
                few_shot_string=flan_fewshot_musique
            else:
                raise ValueError("Bug here, unknown downstream dataset")
            p_prompt_batch = [
                    general_prompt.format(few_shot=few_shot_string,document=example['Proposition'],question=example['Question']) 
                    for example in prompt_batch
                    ]
            tokenized_inputs = tokenizer(p_prompt_batch, padding=True, truncation=True,return_tensors="pt")
            tokenized_inputs["Idx"] = [example["Idx"] for example in prompt_batch]
            batches_tok.append(tokenized_inputs)
        return batches_tok

    accelerator.wait_for_everyone() 

    # save_p = None
    # with accelerator.profile() as prof: 
    
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


        # save_p = prof

    accelerator.wait_for_everyone()  
    # collect results from all the GPUs
    results_gathered=gather_object(results)
    

    if accelerator.is_main_process:
        with open(args.output_path, 'w') as json_file:
            json.dump(results_gathered, json_file, indent=4)

        # print("Profiling flops!")
        # with open("../FLOPS/top5.log","w") as f:
        #     f.write(save_p.key_averages().table(sort_by="flops", row_limit=50))



if __name__ == "__main__":
    main()

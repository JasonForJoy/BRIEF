

import json

datasets_all = ["musique", "2wikimqa", "hotpotqa", "longseal"]

setting = "Brief-Pro_Auto"
# setting = "Brief-Pro_5sentences"
# setting = "Brief-Pro_10sentences"
# setting = "Brief-Pro_20sentences"



import os
import json
from tqdm import tqdm
from openai import OpenAI


# Please enter your OpenAI API Key
client = OpenAI(api_key="Your api_key")




for dataset in datasets_all:

    print(dataset)

    file_path = f"./compdata/{dataset}_unbias_{setting}.jsonl"

    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(len(data))



    answers_file = f"./pred_e/GPT_Results_{setting}/{dataset}_GPT4-1-nano.jsonl"

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")


    for i in tqdm(range(len(data))):

        context = data[i]["context"]
        input = data[i]["input"]

        answers = data[i]["answers"]


        Prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"

        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": Prompt,},
                ],
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=messages,
            temperature=0,
            top_p=1
        )
        



        ans_file.write(json.dumps({
                                "pred": response.choices[0].message.content,
                                "answers": answers,
                                }) + "\n")
        ans_file.flush()



    ans_file.close()



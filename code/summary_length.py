import json
import argparse
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def parse_arg():
    parser = argparse.ArgumentParser(description="Check correctness")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Input path to the json file."
    )

    args = parser.parse_args()
    return args

def count_len(input_path:str,token_level=False,include_empty=False):
    if token_level is True:
        tokenizer = T5Tokenizer.from_pretrained("google/flan-ul2")
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)
    except:
        data = read_jsonl(input_path)
    

    total_word_count = 0

    word_count_list = []
    for item in data:
        proposition = item["reply"]
        if include_empty != True:
            if proposition == "" or "doesn't contain relevant information" in proposition.lower()  or "irrelevant" in proposition.lower():
                continue
        if token_level is True:
            encoded_retrieved_text = tokenizer.encode(proposition, max_length=1000, truncation=True)
            total_word_count += len(encoded_retrieved_text)
            word_count_list.append(len(encoded_retrieved_text))
        elif token_level is False:
            words = proposition.split()
            total_word_count += len(words)
            word_count_list.append(len(words))
 
    # Calculate the average number of words
    average_word_count = total_word_count / len(data) if data else 0
    ave_wc = sum(word_count_list) / len(word_count_list)
    if include_empty:
        assert ave_wc == average_word_count
    
    else:
        print(f"The average number of words(token{token_level}) is: {ave_wc} in {input_path}")

def main():
    args=parse_arg()
    count_len(input_path=args.input_path)

if __name__ == "__main__":
    main()
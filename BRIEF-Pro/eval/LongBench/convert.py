import json
import re
def read_jsonl(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Strip any leading/trailing whitespace and parse the JSON object
                    data.append(json.loads(line.strip()))
            return data
def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
dataset = "musique"
# dataset = "hotpotqa"
# dataset = "2wikimqa"

PATH = f"./data/{dataset}.jsonl"


d = read_jsonl(PATH)
for item in d:
    contexts = item['context']
    pattern = r"Passage \d+:\s*\n(.+?)\n([\s\S]*?)(?=Passage \d+:|\Z)"
    matches = re.findall(pattern, contexts)
    newString = ""
    # Print results
    for idx, (title, content) in enumerate(matches):
        newString += f"Document [{idx}](Title: {title.strip()}){content.strip()}\n"
    item['context'] = newString
    

# save_jsonl(d, PATH.replace("data", "unbiasdata"))
save_jsonl(d, f"./unbiasdata/{dataset}.jsonl")


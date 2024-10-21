
import json
from utils import is_exact_match,normalize_answer,qa_f1_score
from tqdm import tqdm
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description="Check correctness")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path to the json file."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Input path to the json file."
    )
    parser.add_argument(
        "--total_set",
        type=str,
        default=None,
        help="Path to ground truth"
    )
    args = parser.parse_args()
    return args
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
def main():
    args = parse_arg()
    reply_path = args.input_path
    with open(reply_path,"r") as f:
        wiki = json.load(f)
    if "TriviaQA" in args.total_set:
        with open(args.total_set,"r") as f:
            total = json.load(f)
        answer_dict = {item["question_id"]: item["gt_label"] for item in total}
    elif ("NQ" in args.total_set) or ("Hotpot" in args.total_set):
        with open(args.total_set,"r") as f:
            total = json.load(f)
        answer_dict = {item["idx"]: item["gt_label"] for item in total}
    elif ("MuSiQue" in args.total_set):
        with open(args.total_set,"r") as f:
            total = json.load(f)
        answer_dict = {item["question_id"]: [item['gt_label'],item['gt_labels'] ]for item in total}

    new_d = []
    for d in tqdm(wiki):
        if "MuSiQue" in args.total_set or "TriviaQA" in args.total_set:
            question_id = d["QuestionId"]
        else:
            question_id = d["Idx"]

        if "MuSiQue" in  args.total_set:
            single_answer,answer_aliases = answer_dict.get(question_id)
            answer = [single_answer] + answer_aliases
            n_answer = [normalize_answer(a) for a in answer]
            d["correct"] = False
            for ans in n_answer:
                if ans == normalize_answer(d["Proposition_Reply"]):
                    d["correct"] = True
            d["f1_score"]=qa_f1_score(answer,d["Proposition_Reply"])
            new_d.append(d)
        elif "TriviaQA" in args.total_set:
            d["correct"]=is_exact_match(answer,d["Proposition_Reply"])
            d["f1_score"]=qa_f1_score(answer,d["Proposition_Reply"])
            new_d.append(d)
            
        else:
            n_answer = normalize_answer(answer)
            if n_answer == normalize_answer(d["Proposition_Reply"]):
                d["correct"] = True
            d["f1_score"]=qa_f1_score(answer,d["Proposition_Reply"])
            new_d.append(d)


    correct_count = sum(1 for item in new_d if item["correct"])
    accuracy = correct_count / len(new_d)
    mean_f1 = sum(item["f1_score"] for item in new_d) / len(new_d)
    with open(args.output_path,"w") as f:
        json.dump(new_d,f,indent=4)
    with open(args.output_path.replace(".json","_summary.txt"),"w") as f:
        f.write(f"correct_count: {correct_count}\n")
        f.write(f"total_count: {len(new_d)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n")

if __name__ == "__main__":
    main()

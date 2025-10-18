import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
    qa_em_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "longseal_12_docs_noCompressForm": qa_f1_score,
    "longseal_20_docs_noCompressForm": qa_f1_score,
    "longseal_30_docs_noCompressForm": qa_f1_score,
    "longseal_12_docs": qa_f1_score,
    "longseal_20_docs": qa_f1_score,
    "longseal_30_docs": qa_f1_score,
    "hotpotqa_Top5": qa_f1_score,
    "2wikimqa_Top5": qa_f1_score,
    "musique_Top5": qa_f1_score,
    "longseal_12_docs_noCompressForm_Top5": qa_f1_score,
    "hotpotqa_Top1": qa_f1_score,
    "2wikimqa_Top1": qa_f1_score,
    "musique_Top1": qa_f1_score,
    "longseal_12_docs_noCompressForm_Top1": qa_f1_score,
    "hotpotqa_Top10": qa_f1_score,
    "2wikimqa_Top10": qa_f1_score,
    "musique_Top10": qa_f1_score,
    "longseal_12_docs_noCompressForm_Top10": qa_f1_score,
    "hotpotqa_Top3": qa_f1_score,
    "2wikimqa_Top3": qa_f1_score,
    "musique_Top3": qa_f1_score,
    "longseal_12_docs_noCompressForm_Top3": qa_f1_score,
    "musique_dev_Final_500": qa_f1_score,
    "musique_dev_ExpandTop20_Final_500": qa_f1_score,
    "musique_dev_ExpandTop40_Final_500": qa_f1_score,
    "musique_dev_ExpandTop60_Final_500": qa_f1_score,
    "musique_dev_ExpandTop80_Final_500": qa_f1_score
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--short', action='store_true', help="Evaluate on short passages")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    
    # Get rid of left \n
    
    total_score = 0.
    total_em_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        
        prediction = prediction.lstrip('\n')
        score = 0.
        em_score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            em_score = max(em_score, qa_em_score(prediction, ground_truth))
        total_score += score
        total_em_score += em_score
    return round(100 * total_score / len(predictions), 2), round(100 * total_em_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"./pred_e/{args.model}/"
    else:
        path = f"./pred_e/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if args.short:
                    if "length" in data:
                        if (int(data["length"]) < 4000):    
                            lengths.append(data["length"])
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            all_classes = data["all_classes"]
                else:
                    if "length" in data:
                        lengths.append(data["length"])
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    # all_classes = data["all_classes"]


        if args.e:
            # score = scorer_e(dataset, predictions, answers, lengths, all_classes)
            score = scorer_e(dataset, predictions, answers, lengths, None)
        else:
            # score, em_score = scorer(dataset, predictions, answers, all_classes)
            score, em_score = scorer(dataset, predictions, answers, None)
        scores[dataset] = (score, em_score)
    if args.e:
        out_path = f"./pred_e/{args.model}/result.json"
    else:
        if args.short:
            out_path = f"./pred_e/{args.model}/result_short.json"
        else:
            out_path = f"./pred_e/{args.model}/result.json"
        
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

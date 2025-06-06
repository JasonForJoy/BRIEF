import json
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from collections import defaultdict
from tqdm import tqdm
import math
from collections import defaultdict
import numpy as np

def sorted_dict(dict:dict[str,list[list]]):
    """ sorted_dict takes in a dict whose value is a list of sublists. This function rerank the list according to the certain value of its sublists, to be specific, the second element.
    """
    aggregated_dict = {}
    for key, value in dict.items():
        question_score_map = defaultdict(list)
        for sentence, score in value:
            try:
                question_score_map[sentence].append(score)
            except:
                print(value)
        
        aggregated_list = []
        for sentence, scores in question_score_map.items():
            mean_score = np.mean(scores)
            aggregated_list.append([sentence, mean_score])
        
        # Sort by mean score in descending order
        aggregated_list.sort(key=lambda x: x[1], reverse=True)
        
        aggregated_dict[key] = aggregated_list
    
    return aggregated_dict

def aggregate_scores(dict1:dict[str,list[list]], dict2:dict[str,list[list[str,float]]]):
    """ aggregate_scores takes in 2 dict and return a combined dict.

    Note:
        The dict's value is a list of list. The sublist contains the sentence in idx 0 and the improvement in idx 1. Call this function and you will get a combined dict whose value is still a list of list, but with sorted order with average of give 2 dicts.
    """
    combined_dict = defaultdict(list)
    sentence_presence = defaultdict(lambda: [False, False])
    
    # Combine the dictionaries and track presence
    for i, d in enumerate([dict1, dict2]):
        for key, value in d.items():
            for sentence, score in value:
                combined_dict[key].append([sentence, score])
                sentence_presence[sentence][i] = True
    
    # Calculate the adjusted scores
    aggregated_dict = {}
    for key, value in combined_dict.items():
        question_score_map = defaultdict(list)
        for sentence, score in value:
            if sentence_presence[sentence] == [True, False] or sentence_presence[sentence] == [False, True]:
                question_score_map[sentence].append(score / 2)
            else:
                question_score_map[sentence].append(score)
        
        aggregated_list = []
        for sentence, scores in question_score_map.items():
            mean_score = np.mean(scores)
            aggregated_list.append([sentence, mean_score])
        
        # Sort by mean score in descending order
        aggregated_list.sort(key=lambda x: x[1], reverse=True)
        
        aggregated_dict[key] = aggregated_list
    
    return aggregated_dict


def visualize_distribution(file:str,field:str="Improvement",img_name:str='likelihood_distribution.png'):
    """ visualize_distribution takes in a file, analyze the distribution of a given field. 

    Args:
        file:     A string that represents the path to the input file
        field:    A string that represents the field that will be counted to get the distribution
        img_name: A string that tells us the save path of the output image

    Note:
        We assume that the file is a list of dicts with at least key `field` and `QuestionId`. The `QuestionId` matters because we are actually getting the max_`field` distribution across different questions. This `QuestionId` can help us locate different items under the same question. 
    """
    path = file
    with open(path,"r") as f:
        wiki = json.load(f)
    d = defaultdict(list)
    for item in wiki:
        d[item["QuestionId"]].append(item[field])
    best_logit = [max(v) for v in d.values()]
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    percentile_25 = np.percentile(best_logit, 25)
    percentile_50 = np.percentile(best_logit, 50)
    percentile_10 = np.percentile(best_logit, 10)
    plt.figure(figsize=(10, 6))
    sns.histplot(best_logit, bins=100, kde=True)
    plt.axvline(percentile_50, color='g', linestyle='--', label=f'50th Percentile (Median): {percentile_50:.2f}')
    plt.axvline(percentile_10, color='b', linestyle='--', label=f'10th Percentile: {percentile_10:.2f}')
    plt.axvline(percentile_25, color='r', linestyle='--', label=f'25th Percentile: {percentile_25:.2f}')
    # Add text annotations for the percentiles
    plt.text(percentile_50, plt.ylim()[1]*0.9, f'{percentile_50:.2f}', color='g', ha='center')
    plt.text(percentile_10, plt.ylim()[1]*0.9, f'{percentile_10:.5f}', color='b', ha='center')
    plt.text(percentile_25, plt.ylim()[1]*0.9, f'{percentile_25:.2f}', color='r', ha='center')
    plt.title(f'Distribution of {field}')
    plt.xlabel('Best Logit')
    plt.ylabel('Frequency')
    plt.savefig(f'./{img_name}')
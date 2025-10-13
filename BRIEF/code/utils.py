import string
import re
from tqdm import tqdm
from collections import Counter

# From TriviaQA evaluation script
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
        return text.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def get_ground_truths(answer):
    return answer['NormalizedAliases'] + [normalize_answer(ans) for ans in answer.get('HumanAnswers', [])]

def is_exact_match(answer_object, prediction):
    # If multiple ground truths, then we return True if at least one is an exact match
    # This means we return the maximum exact match score
    ground_truths = get_ground_truths(answer_object)
    for ground_truth in ground_truths:
        if exact_match_score(prediction, ground_truth):
            return True
    return False


# From LLMLingua & Compact script
def f1_score( ground_truth,prediction):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(ground_truth,prediction):
    # This is for TriviaQA, which provides multiple ground truths
    # In this case, we return the maximum F1 score
    if isinstance(ground_truth,dict) and "NormalizedAliases" in ground_truth:
        ground_truths = get_ground_truths(ground_truth)
        all_f1 = [qa_f1_score(ground_truth,prediction) for ground_truth in ground_truths]
        return max(all_f1)
    # If it's a list, then it's a list of ground truths
    elif isinstance(ground_truth,list):
        all_f1 = [qa_f1_score(gt,prediction) for gt in ground_truth]
        return max(all_f1)
    # If it's a string, then it's a single ground truth
    elif isinstance(ground_truth,str):
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        return f1_score(ground_truth_tokens,prediction_tokens)


def judge_correctness(wiki):
    new_list_of_dict = []
    for question_dict in tqdm(wiki):
        if is_exact_match(question_dict['Answer'],question_dict['Proposition_Reply']):
            new_list_of_dict.append({"Question":question_dict['Question'],"Proposition":question_dict['Proposition'],"Passage":question_dict["Passage"],"QuestionId":question_dict["QuestionId"]})
    return new_list_of_dict
import string
import re
from tqdm import tqdm
import pandas as pd
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

def extend_passage(passage, previous_passage="", next_passage="",view=50):
    """Extend the passage with a bit of context from previous and next passages."""
    return f"{previous_passage[-view:]} {passage} {next_passage[:view]}"


# From LLMLingua & Compact
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

# Comment out to use this function


# import spacy
# def ensure_complete_sentences(passage, original_passage):
#     """Ensure the passage contains complete sentences using sentence boundary detection."""
#     # Load SpaCy model for sentence boundary detection
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(passage)
#     sentences = [sent.text for sent in doc.sents]

#     # Find the index range of the original passage within the extended passage
#     original_start = passage.find(original_passage)
#     original_end = original_start + len(original_passage)
#     # Filter sentences to keep only those that fall within the original passage boundaries
#     complete_sentences = []
#     for i,sent in enumerate(sentences):
#         sent_start = passage.find(sentences[i])
#         sent_end = sent_start + len(sentences[i])
#         if not ((sent_start <= original_start and sent_end <= original_start) or (sent_start >= original_end and sent_end >= original_end)):
#             complete_sentences.append(sent)

#     return " ".join(complete_sentences)

def preprocess_passages(df,apply_to_all:bool=False):
    """Preprocess the passages in the DataFrame to handle split sentences and add a new column."""
    if apply_to_all:
        name_prefix = "Top_5_"
    else:
        name_prefix = ""
    preprocessed_passages = []
    for idx, row in tqdm(df.iterrows()):
        preprocessed_row_passages = []
        for is_correct, passage in zip(row['Passage_Correct'], row['Top_5_Passage']):
            is_correct = (apply_to_all or is_correct)
            if is_correct:
                # Find the index of the selected passage in the list of passages
                passage_index = row['Passages'].index(passage)
                
                # Retrieve previous and next passages from the list of passages
                previous_passage = row['Passages'][passage_index - 1] if passage_index > 0 else ""
                next_passage = row['Passages'][passage_index + 1] if passage_index < len(row['Passages']) - 1 else ""
                
                # Extend and ensure complete sentences
                extended_passage = extend_passage(passage, previous_passage, next_passage,view=150)
                complete_passage = ensure_complete_sentences(extended_passage,passage)
                
                preprocessed_row_passages.append(complete_passage)
        
        preprocessed_passages.append(preprocessed_row_passages)
    
    # Add the new column to the DataFrame
    df.loc[:, f'{name_prefix}Preprocessed_Passages'] = pd.Series(preprocessed_passages)
    return df

def drop_column(df,columns:list[str]=[]):
    df = df.drop(columns=columns)
    return df

def judge_correctness(wiki):
    new_list_of_dict = []
    for question_dict in tqdm(wiki):
        if is_exact_match(question_dict['Answer'],question_dict['Proposition_Reply']):
            new_list_of_dict.append({"Question":question_dict['Question'],"Proposition":question_dict['Proposition'],"Passage":question_dict["Passage"],"QuestionId":question_dict["QuestionId"]})
    return new_list_of_dict
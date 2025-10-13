import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from custom_utils import is_exact_match
import json
import pandas as pd
from tqdm import tqdm
from Inference_utils.inference_engine import reply_engine,embed_engine,propositionizer
from Inference_utils.wait_until_free import wait_until_free_do
from collections import namedtuple
import math
import logging
###############################################
###############################################
# region Step 0 Datastructure Specification
# We need a data structure that can be used to communicate through different files

CONFIG = {
    "embedding_model":"contriever",
    "reply_model":"flan-ul2",  # alternative, llama3 flan-ul2
    }
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("pipeline.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
# endregion

###############################################
###############################################
# region Step 1 Read in data and get plain reply and prompt

TriviaQA = namedtuple("TriviaQA",["Answer","EntityPages","Question"])

# TriviaQA
with open("trivia_qa_data/qa/wikipedia-train.json","r") as f:
    wiki = json.load(f)
wiki = wiki['Data']
wiki:dict[str,TriviaQA] = {w["QuestionId"]:TriviaQA(w["Answer"],w["EntityPages"],w["Question"]) for w in wiki}

# NQ
# with open("NQ/nq_find_top_5.json","r") as f:
#     wiki = json.load(f)
# wiki:dict[str,TriviaQA] = {w["QuestionId"]:TriviaQA(w["Answer"],w["EntityPages"],w["Question"]) for w in wiki}

# Remove the propositions to get the plain reply and prompt
with open("./inpath_no_proposition.json","w") as f:
    dump = [{"QuestionId":idx,"Question":d.Question,"Proposition":""} for idx,d in wiki.items()]
    json.dump(dump,f,indent=4)
logger.info("Now prepare to reply none-proposition")
reply_engine(reply_model=CONFIG["reply_model"],inpath="./inpath_no_proposition.json",outpath="./outpath_no_proposition_fix.json")
logger.info("Finish reply none-proposition")
# endregion

###############################################
###############################################
# region Step 2 Get 100-doc reply and prompt


with open("./inpath_find_top_5.json","w") as f:
    dump = [{"QuestionId":k,**v._asdict()} for k,v in wiki.items()]
    json.dump(dump,f,indent=4)


logger.info("Now prepare to find top 5 passages")
embed_engine(wiki_split="psgs_w100.tsv",inpath="./inpath_find_top_5.json",outpath="./outpath_find_top_5.json",top_k=5)
logger.info("Finish find top 5 passages")

with open("./outpath_find_top_5.json","r") as f:
    dump = json.load(f)
with open("./inpath_100_word_proposition.json","w") as f:
    dump = [{"QuestionId":d["QuestionId"],"Question":d["Question"],"Proposition":passage} for d in dump for passage in d["Top_Passages"] ]
    json.dump(dump,f,indent=4)


logger.info("Now prepare to reply 100-word-proposition") 
w_reply_engine = wait_until_free_do(reply_engine)  
w_reply_engine(reply_model=CONFIG["reply_model"],inpath="inpath_100_word_proposition.json",outpath="flan_ul2_reply_100_word_proposition.json",batch_size="2",cuda_setting="0,1,2,3",instruct=False)
logger.info("Finish reply 100-word-proposition")


# endregion

###############################################
###############################################
# region Step 2.1 Identify which 100-doc is helpful
# For ul2, it's Sequence_Probability
# For llama, it's Len_Seq_Probs
model_field_dict = {"flan":"Sequence_Probability","llama":"Len_Seq_Probs"}
def identify_useful_docs(list1:list[dict], list2:list[dict],model:str = "llama"):
    
    # Create a mapping of QuestionId to entries in list1
    list1_map = {doc['QuestionId']: doc for doc in list1}
    useful_docs = []
    for doc2 in tqdm(list2):
        question_id = doc2['QuestionId']
        if question_id in list1_map:
            doc1 = list1_map[question_id]
            
            reply1_correct:bool = is_exact_match(wiki.get(question_id).Answer,doc1['Proposition_Reply'])
            reply2_correct:bool = is_exact_match(wiki.get(question_id).Answer,doc2['Proposition_Reply'])
            
            if not reply1_correct and reply2_correct:
                # Case 1: Reply with null string is incorrect, and reply with proposition is correct
                if doc2[model_field_dict.get(model)] != None:
                    improvement = math.exp(doc2[model_field_dict.get(model)])
                    useful_docs.append({
                        **doc2,
                        'Improvement': improvement
                    })
            elif reply1_correct and reply2_correct:
                # Case 2: Both replies are correct, compare Sequence_Probabilities
                if doc1[model_field_dict.get(model)] != None and doc2[model_field_dict.get(model)] != None:
                    if doc2[model_field_dict.get(model)] > doc1[model_field_dict.get(model)]:
                        improvement = math.exp(doc2[model_field_dict.get(model)]) - math.exp(doc1[model_field_dict.get(model)])
                        useful_docs.append({
                            **doc2,
                            'Improvement': improvement
                        })
    
    return useful_docs

logger.info("Now we are detecting useful documents.")
with open("flan_ul2_no_proposition.json","r") as f:
    none_proposition_wiki = json.load(f)

with open("flan_ul2_propositions_batch_result.json","r") as f:
    per_prop_wiki = json.load(f)
useful_docs = identify_useful_docs(none_proposition_wiki,per_prop_wiki,model="flan")
out_path = "flan_ul2_useful_propositions.json"
with open(out_path,"w") as f:
    json.dump(useful_docs,f,indent=4)

logger.info(f"We save the useful docs to the file: {out_path}")


# Recover & Proposition
# Simply call `recover_split_to_complete.py` to finish the recover step. You should get a `json` file with prefix `complete`

# Now let's assume we have `complete_flanul2_useful_docs.json`
# The next thing we need to do is to propositionize them.
# We need to run `accelerate_propositionize.py` to do this

propositionizer(batch_size="64",inpath="complete_llama_useful_docs.json",outpath="llama_propositions.json",cuda_setting="0,1,2,3")


# endregion


# Step 3 Decontextualization
# We need to decontextualize the propositions
# For the propositions generated by Dense X Retrieval has its own limits due to incomplete 100-word limitation
# Run `src/Decontextualize/async_decontextualize.py` to do this.



# Step 4
# Per-proposition helpfulness assessment
# In this step, we will label whether a proposition is helpful or not
# The input is a list of dict, each dict has the following format:
# {
#         "QuestionId": "tc_3",
#         "Question": "Where in England was Dame Judi Dench born?",
#         "Proposition": "Judi Dench attended the Mount School.",
#         "Idx": 0
#     },
logger.info("Begin reply propositions")

def mod(l:list):
    return l + ["--test"]
import time
import sys
# waitTime = int(3600*2)
waitTime = int(35*60)
# waitTime = 0
for remaining in range(waitTime, 0, -1):
    sys.stdout.write("\r")
    sys.stdout.write("{:2d} seconds remaining.".format(remaining))
    sys.stdout.flush()
    time.sleep(1)


reply_engine = wait_until_free_do(reply_engine,gpu_lists=[4,5,6,7])



reply_engine(reply_model=CONFIG["reply_model"],inpath="usedoc_decomp.json",outpath="llama_result.json",batch_size="4",cuda_setting="4,5,6,7",instruct="True")

logger.info("Finish reply propositions")

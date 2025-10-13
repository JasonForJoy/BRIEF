"""
This script is used to embed the wiki dump and the questions into embeddings using the Contriever model.

`Ctrl+F` to search for `DONOT` to find the parts that may need to be modified for different datasets.
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
# from prompt import prompt
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "3600"  # Set NCCL timeout to 1 hour (3600 seconds)
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

def arg_parse():
    parser = argparse.ArgumentParser(description="Inference a Flan UL2 using Accelerate")
    parser.add_argument(
        "--wiki_split",
        type=str,
        default="./wiki_dump/psgs_w100.tsv",
        help="Library to find wiki dump"
        )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top 5 wiki dump"
        )
    parser.add_argument(
        "--inpath",
        type=str,
        default=None,
        help="Input path to the json file. Must be a list of dicts."
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Output path to the json file. Will be a list of dicts."
    )
    args=parser.parse_args()
    return args
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
def main():
    args = arg_parse()
    model_name = "facebook/contriever-msmarco"
    dir = "facebook/contriever-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(dir, local_files_only=True)
    model = AutoModel.from_pretrained(dir, local_files_only=True)

    # Check for GPU availability and wrap the model with DataParallel
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        print("Using GPUs:", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("Using CPU")

    

    def get_sentence_embed(sentences: list[str], batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            def mean_pooling(token_embeddings, mask):
                token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                return sentence_embeddings
            batch_embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings.append(batch_embeddings.cpu())
        return torch.cat(embeddings, dim=0)

    def preprocess_embeddings(df: pd.DataFrame, relevant_titles: set, batch_size=128):
        embeddings_dict = {}
        filtered_df = df[df['title'].isin(relevant_titles)]
        print(len(filtered_df))
        passages = filtered_df['text'].tolist()
        titles = filtered_df['title'].tolist()
        embeddings = get_sentence_embed(passages, batch_size=batch_size).numpy()

        for title, embedding in zip(titles, embeddings):
            # print(f"title: {title}")
            if title not in embeddings_dict:
                embeddings_dict[title] = []
            embeddings_dict[title].append(embedding)

        return embeddings_dict

    def make_prompt(wiki, df=None, k=5, initial_batch_size=32):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract relevant titles from the wiki data
        relevant_titles = set(doc['Title'] for question in wiki for doc in question['EntityPages'])
        print(f"len(relevant_titles): {len(relevant_titles)}")
        print(f"len(wiki,old): {len(wiki)}")
        
        
        # Precompute embeddings for all relevant passages in the DataFrame
        embeddings_dict = preprocess_embeddings(df, relevant_titles)


        # This assumes that each question has only one entity page
        # Applies only to NQ, DONOT USE FOR OTHER DATASETS like TriviaQA
        # wiki = [item for item in wiki if item['EntityPages'][0]['Title'] in embeddings_dict]


        # print(embeddings_dict.keys())
        print(f"len(wiki,new): {len(wiki)}")
        # Prepare all questions for batch embedding
        print("Now we are preparing all questions for batch embedding")
        questions = [question_dict['Question'] for question_dict in wiki]
        query_embeddings = get_sentence_embed(questions).numpy()

        current_batch_size = initial_batch_size
        print("Now we are processing each question in batches and retrieve top-k passages")
        print(f"len(wiki): {len(wiki)}, current_batch_size: {current_batch_size}")
        # Process each question in batches and retrieve top-k passages
        for i in tqdm(range(0, len(wiki), current_batch_size)):
            batch_wiki = wiki[i:i+current_batch_size]
            batch_query_embeddings = query_embeddings[i:i+current_batch_size]
            # If we want to search the Top-5 passages in the scope of all batch questions
            # This method shows no obvious issues, but the results may not be as good as the following method
            q_passages = []
            q_passage_embeddings = []
            for idx, question_dict in enumerate(batch_wiki):
                # If we only search in the scope of the current question 
                # Use with caution, a prototype running shows that it may have some issues
                # Specifically, the top-5 passages retrieved for each question may be quite different from the actual top-5 passages
                # DONOT USE THIS unless you know what you are doing
                # q_passages = []
                # q_passage_embeddings = []

                wiki_docs = question_dict['EntityPages']
                for wiki_doc in wiki_docs:
                    title = wiki_doc['Title']
                    # print(f"Outside title: {title}")
                    if title in embeddings_dict:
                        # print(f"title: {title}")
                        current_title_df = df[df['title'] == title]
                        passages = current_title_df['text'].tolist()
                        # print(f"passages: {passages[:1]}")
                        passage_embeddings = embeddings_dict[title]
                        q_passages.extend(passages)
                        q_passage_embeddings.extend(passage_embeddings)
                
            if len(q_passage_embeddings) < initial_batch_size:
                raise ValueError(f"{len(q_passage_embeddings)}")


            if len(q_passage_embeddings) >0:
                
                q_passage_embeddings = np.array(q_passage_embeddings)
                q_passage_embeddings = torch.tensor(q_passage_embeddings).to(device)
                # print(f"q_passage_embeddings: {q_passage_embeddings.shape}")
                batch_query_embeddings = torch.tensor(batch_query_embeddings).to(device)
            else:
                # question_dict['Top_Passages'] = []
                raise ValueError("No passages found!")

            for idx, question_dict in enumerate(batch_wiki):
                try:
                    # Calculate cosine similarity for this question on the GPU
                    query_embedding = batch_query_embeddings.unsqueeze(1)[idx]
                    similarities = torch.nn.functional.cosine_similarity(
                        query_embedding, q_passage_embeddings.unsqueeze(0), dim=2
                    )

                    # Retrieve top-k passages for this query 
                    # print(len(similarities))
                    
                    # for iidx,query_similarity in enumerate(similarities):
                    query_similarity = similarities[0]
                    top_k_indices = torch.topk(query_similarity, len(query_similarity)).indices
                    top_k_passages = remove_duplicates([q_passages[i] for i in top_k_indices])[:k]
                    question_dict['Top_Passages'] = top_k_passages
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"Out of memory, reducing batch size to {current_batch_size // 2}")
                        current_batch_size = max(1, current_batch_size // 2)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(e)
                        continue
            
            
            # This use with the outer loop
            # if q_passage_embeddings is not None:
            #     q_passage_embeddings = np.array(q_passage_embeddings)
            #     q_passage_embeddings = torch.tensor(q_passage_embeddings).to(device)
            #     batch_query_embeddings = torch.tensor(batch_query_embeddings).to(device)

            #     try:
            #         # Calculate cosine similarity for the entire batch on the GPU
            #         similarities = torch.nn.functional.cosine_similarity(
            #             batch_query_embeddings.unsqueeze(1), q_passage_embeddings.unsqueeze(0), dim=2
            #         )

            #         # Retrieve top-k passages for each query in the batch
            #         for idx, question_dict in enumerate(batch_wiki):
            #             query_similarities = similarities[idx]
            #             top_k_indices = torch.topk(query_similarities, min(k, len(query_similarities))).indices
            #             top_k_passages = [q_passages[i] for i in top_k_indices]
            #             question_dict['Top_Passages'] = top_k_passages
            #     except RuntimeError as e:
            #         if 'out of memory' in str(e):
            #             print(f"Out of memory, reducing batch size to {current_batch_size // 2}")
            #             current_batch_size = max(1, current_batch_size // 2)
            #             torch.cuda.empty_cache()
            #             continue
            #         else:
            #             print(e)
            #             continue
            # else:
            #     for question_dict in batch_wiki:
            #         question_dict['Top_Passages'] = []

        return wiki

    with open(args.inpath, "r") as f:
        wiki = json.load(f)

    wiki = wiki[:100]

    path = args.wiki_split
    df = pd.read_csv(path, sep='\t')
        # For testing
        # DONOT USE THIS when the input is the whole dataset
        # wiki = wiki[:10]
    wiki_augment_prompt = make_prompt(wiki, df=df, k=args.top_k,initial_batch_size=256)
    with open(args.outpath, "w") as f:
        json.dump(wiki_augment_prompt, f, indent=4)

if __name__ == "__main__":
    main()

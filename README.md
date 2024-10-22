<div align="center">

# BRIEF: Bridging Retrieval and Inference for Multi-hop Reasoning via Compression

[Yuankai Li*](https://lieziwind.github.io/), [Jia-Chen Gu*](https://jasonforjoy.github.io/), [Di Wu](https://xiaowu0162.github.io/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/) and [Nanyun Peng](https://violetpeng.github.io/)

*Equal contribution

<a href='https://arxiv.org/pdf/2410.15277'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://jasonforjoy.github.io/BRIEF/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<!-- <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href=''><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-ffd21e"></a> <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> <a href=''><img src='https://img.shields.io/badge/Project-Page-Green'></a>  -->

</div>

This repository hosts the code and data for our paper **BRIEF**, a lightweight, T5-based approach that performs query-aware multi-hop reasoning by compressing retrieved documents into highly dense textual summaries to integrate into in-context learning.

## Overview

BRIEF (Bridging Retrieval and Inference through Evidence Fusion) is a lightweight approach that performs query-aware multi-hop reasoning by compressing retrieved documents into highly dense textual summaries to integrate into in-context learning. To enable learning compression for multi-hop reasoning, we curate synthetic data by extracting atomic proposition expressions that encapsulate distinct factoids from the source documents to compose synthetic summaries.

<p align="center">
  <img src="src/BRIEF_inference.png" width="100%"></a> <br>
</p>

<p align="center">
  <img src="src/BRIEF_train.png" width="100%"></a> <br>
</p>
BRIEF generates more concise summaries and enables a range of LLMs to achieve exceptional open-domain question answering (QA) performance. For example, on HotpotQA, BRIEF improves the compression rate by 2 times compared to the state-of-the-art baseline, while outperforming it by 3.00% EM and 4.16% F1 with Flan-UL2 as the reader LM. It also generates more concise summaries than proprietary GPT-3.5, while demonstrating nearly identical QA performance.
<p align="center">
  <img src="src/result_multihop.png" width="100%"></a> <br>
</p>

## Release

- [10/21]  We released the compressed result and evaluation script for TriviaQA, NQ, HotpotQA and MuSiQue.

## Installation and Setup

1. Clone this repository and navigate to the BRIEF folder
   
   ```bash
   git clone https://github.com/JasonForJoy/BRIEF
   cd BRIEF/code
   ```

2. Install Package
   
   ```bash
   conda create -n brief python=3.9 -y
   conda activate brief
   pip install --upgrade pip  
   pip install -e .
   ```
   
   If this doesn't work, just install the newest version of `pytorch`, `transformers` and `accelerate`

## Start with FlanUL2

To adjust `accelerate` to your needs, you may `accelerate config` first. Then, the code below shows how to inference through FlanUL2 using the summarized documents. The following is an example for TriviaQA.

<!-- <details>
<summary>Example Code</summary> -->

```bash
accelerate launch --main_process_port 29500 flanul2_reader.py \
--inference_type ours \
--proposition_name_or_path ../data/TriviaQA_brief_reply.json \
--output_path TriviaQA_read.json \
--downstream_dataset tqa
```

## Evaluation Code

When evaluating the downstream performance of our model, simply follow the code below.

<!-- <details>
<summary>Example Code</summary> -->

```bash
python evaluation.py --total_set ../data/ground_truth/TriviaQA_GT.json \
--input_path TriviaQA_read.json
--output_path TriviaQA_result.json
```

You can also check the length of the summary.

<!-- <details>
<summary>Example Code</summary> -->

```bash
python summary_length.py --input_path ../data/TriviaQA_brief_reply.json
```

## Baselines of other models

- RECOMP
  
  - Follow [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://github.com/carriex/recomp/tree/main)

- LLMLingua
  
  - Follow the example at [LLMLingua/examples/RAG.ipynb](https://github.com/microsoft/LLMLingua/blob/main/examples/RAG.ipynb)

- Selective-Context
  
  - Follow [Selective Context for LLMs](https://github.com/liyucheng09/Selective_Context)
  
  - Note: we explicitly use `python==3.9.19, typing_extensions==4.8.0, thinc==8.0.17, spacy==3.2.0, pydantic==1.7.4, torch==2.4.1, numpy==1.26.4` since the original `requirements.txt` is broken.

## Citation

If you find BRIEF useful for your research and applications, please cite using this BibTeX:

```bibtex
@article{li2024brief,
 title = "BRIEF: Bridging Retrieval and Inference for Multi-hop Reasoning via Compression",
 author = "Li, Yuankai  and
           Gu, Jia-Chen  and
           Wu, Di  and
           Chang, Kai-Wei  and
           Peng, Nanyun",
 journal={arXiv preprint arXiv:2410.15277},
 year = "2024"
}
```

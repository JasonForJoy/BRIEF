# BRIEF-PRO: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning

[Jia-Chen Gu*](https://jasonforjoy.github.io/), [Junyi Zhang*](https://levi-zjy.github.io/), [Di Wu](https://xiaowu0162.github.io/), [Yuankai Li](https://lieziwind.github.io/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/) and [Nanyun Peng](https://violetpeng.github.io/)


[![License](BRIEF-Pro/src/license.svg)]()
[![Python](BRIEF-Pro/src/python38.svg)]()
<!-- [![Stars](BRIEF-Pro/src/stars.svg?style=social)]() -->


<!-- [**🌐 Homepage**]() | [**🤗 Dataset**](https://huggingface.co/datasets/uclanlp/Brief-Pro) | [**🤗 Model**](https://huggingface.co/uclanlp/brief-pro) | [**📖 Paper**](https://arxiv.org/abs/2510.13799) -->


🔗 **Paper**: https://arxiv.org/abs/2510.13799

🌐 **Website**: 

🤗 **Dataset:** https://huggingface.co/datasets/uclanlp/Brief-Pro

🤗 **Model:** https://huggingface.co/uclanlp/brief-pro




## 🚀 Overview

🤖 **BRIEF-PRO** is a universal, lightweight compressor that distills relevant evidence for a given query from multiple retrieved documents into a concise summary for seamless integration into in-context RAG.

<p align="left">
  <img src="BRIEF-Pro/src/Teaser_fig.png" width="50%"></a> <br>
</p>

## News

- We released the training and evaluation code and the training data.

## Intro


- BRIEF-PRO is a universal, lightweight compressor that distills relevant evidence for a given query from multiple retrieved documents into a concise summary for seamless integration into in-context RAG.
- Given seed data with contexts that are relatively short, each containing fewer than 1k words, BRIEF-PRO is trained towards abstractively condensing long contexts of 10k+ words across diverse scenarios.
- Furthermore, BRIEF-PRO offers flexible user control over the length of the output summary by allowing users to specify the desired number of sentences.


<p align="center">
  <img src="BRIEF-Pro/src/Teaser_fig.png" width="60%"></a> <br>
</p>


- The overview of the synthetic data pipeline for training BRIEF-PRO is as follows. Starting with a mixture of oracle and distractor documents for a given query, the pipeline: (a) expands each document by looking up an external knowledge corpus, (b) curates a compact summary by further reducing redundancy in the oracle documents, and (c) generates a user-controllable compression instruction by counting the number of sentences in the compact summary.


<p align="center">
  <img src="BRIEF-Pro/src/Pipline.png" width="100%"></a> <br>
</p>





## Training 

- Follow the instructions on the [Axolotl website](https://github.com/axolotl-ai-cloud/axolotl) to set up the training environment.

- Then run the following command to start training:


  ```
  bash ./BRIEF-Pro/train/Axolotl/examples/llama-3.2/test3.sh
  ```

## Evaluation 

- Run the following command to for evaluation:
  

  ```
  bash ./BRIEF-Pro/BRIEF/BRIEF-Pro/eval/test_pipe_testAll_UserControl.sh
  bash ./BRIEF-Pro/BRIEF/BRIEF-Pro/eval/test_pipe_testSealQA_UserControl.sh
  ```





## Citation


# 🤖 BRIEF-PRO: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning

**[Jia-Chen Gu*](https://jasonforjoy.github.io/), [Junyi Zhang*](https://levi-zjy.github.io/), [Di Wu](https://xiaowu0162.github.io/), [Yuankai Li](https://lieziwind.github.io/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/) and [Nanyun Peng](https://violetpeng.github.io/)**


[![License](BRIEF-Pro/src/license.svg)]()
[![Python](BRIEF-Pro/src/python38.svg)]()
<!-- [![Stars](BRIEF-Pro/src/stars.svg?style=social)]() -->


<!-- [**🌐 Homepage**]() | [**🤗 Dataset**](https://huggingface.co/datasets/uclanlp/Brief-Pro) | [**🤗 Model**](https://huggingface.co/uclanlp/brief-pro) | [**📖 Paper**](https://arxiv.org/abs/2510.13799) -->


🔗 **Paper**: https://arxiv.org/abs/2510.13799

🌐 **Website**: 

🤗 **Dataset:** https://huggingface.co/datasets/uclanlp/Brief-Pro

🤗 **Model:** https://huggingface.co/uclanlp/brief-pro



## News

🔥 We released the training and evaluation code, the model checkpoint, and the training data.




## 🚀 Overview

🤖 **BRIEF-PRO** is a universal, lightweight compressor that distills relevant evidence for a given query from multiple retrieved documents into a concise summary for seamless integration into in-context RAG.

<p align="left">
  <img src="BRIEF-Pro/src/Teaser_fig.png" width="50%"></a> <br>
</p>


## ✨ Key Features

- 🔧 BRIEF-PRO pioneers the exploration of multi-hop reasoning and compression of RAG for long contexts of 10k+ words across diverse scenarios. 
- ⚙️ A synthetic data pipeline, built on short-context seed data, is designed to synthesize long-context training data for compression learning.
- 🧩 BRIEF-PRO, trained on the curated dataset, generates concise summaries that accelerate the inference and enhance the accuracy of a wide range of small, large, and proprietary language models.

<p align="center">
  <img src="BRIEF-Pro/src/Pipline.png" width="90%"></a> <br>
</p>


## 📍 Installation

- [TBD]




## 🏃🏻‍♀️ Training 

- Follow the instructions on the [Axolotl website](https://github.com/axolotl-ai-cloud/axolotl) to set up the training environment.

- Then run the following command to start training:


  ```
  bash ./BRIEF-Pro/train/Axolotl/examples/llama-3.2/test3.sh
  ```

## 🔬 Evaluation 

- Run the following command to for evaluation:
  

  ```
  bash ./BRIEF-Pro/BRIEF/BRIEF-Pro/eval/test_pipe_testAll_UserControl.sh
  bash ./BRIEF-Pro/BRIEF/BRIEF-Pro/eval/test_pipe_testSealQA_UserControl.sh
  ```





## Citation


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




<!-- 
## 📍 Installation

-  
-->




## 🏃🏻‍♀️ Training 

**📍 Installation:**

<!-- Follow the instructions on the [Axolotl website](https://github.com/axolotl-ai-cloud/axolotl) to set up the training environment. -->

```
conda create -n axolotl python=3.10 -y
conda activate axolotl

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install packaging ninja
pip install flash-attn --no-build-isolation

cd Axolotl/
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
pip install xformers==0.0.29.post2
pip install axolotl==0.5.0 accelerate peft optimum bitsandbytes liger-kernel lm-eval

pip install -e .
pip install -e '.[deepspeed]'
```

You can also follow the instructions on the [Axolotl website](https://github.com/axolotl-ai-cloud/axolotl) (we use axolotl==0.5.0) to set up the training environment.


**💡 Running:**


Then run the following command to start training:


```
bash ./BRIEF-Pro/train/Axolotl/examples/llama-3.2/train.sh
```



## 🔬 Evaluation 

**📍 Installation:**

Follow [VLLM](https://github.com/vllm-project/vllm) to install the **multidoc_vllm** environment.

Follow [LongBench](https://github.com/THUDM/LongBench) to install the **longbench** environment.

You can also quickly set up the environments using the provided .yml files.
```
conda env create -f ./BRIEF-Pro/env/multidoc_vllm_env.yml
conda env create -f ./BRIEF-Pro/env/longbench_env.yml
```




**💡 Running:**

Run the following command to for evaluation:
  
- BRIEF-PRO as the Compressor 
  - Llama-3.1-8B-Instruct / Llama-3.1-70B-Instruct as the Reader Model:
  ```
  bash ./BRIEF-Pro/eval/test_pipe_testAll_UserControl.sh
  bash ./BRIEF-Pro/eval/test_pipe_testSealQA_UserControl.sh
  ```

  - GPT-4.1-nano as the Reader Model:
  ```
  python ./BRIEF-Pro/eval/GPT_pred.py
  ```





## Citation

```
@misc{gu2025briefprouniversalcontextcompression,
      title={BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning}, 
      author={Jia-Chen Gu and Junyi Zhang and Di Wu and Yuankai Li and Kai-Wei Chang and Nanyun Peng},
      year={2025},
      eprint={2510.13799},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.13799}, 
}
```


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


from tqdm import tqdm


# ===================== I/O 工具 =====================

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


# ===================== 核心：取最后 token 的隐藏态 =====================

"""
@torch.no_grad()
def last_token_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    
    # 对每个 prompt：
    #   - 编码（padding 到 batch 内最大长度，truncation 开启，且不再添加额外 special tokens）
    #   - 前向计算并开启 output_hidden_states
    #   - 取最后一层隐藏态在每条样本“最后一个非 padding token”的向量
    # 返回：形状 [N, H] 的 numpy 数组
    
    device = next(model.parameters()).device
    all_vecs = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            # padding=True,
            padding=False,
            # truncation=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # 最后一层隐藏态：[B, T, H]
        last_layer = out.hidden_states[-1]
        # 每条样本的“最后一个非 padding 位置”
        last_idx = attention_mask.sum(dim=1) - 1  # [B]
        b_idx = torch.arange(last_layer.size(0), device=device)
        last_vec = last_layer[b_idx, last_idx]    # [B, H]

        print(last_vec)

        all_vecs.append(last_vec.detach().cpu())

    hs = torch.cat(all_vecs, dim=0)  # [N, H]

    # return hs.numpy()
    return hs.to(torch.float32).numpy()
"""


@torch.no_grad()
def last_token_hidden_states_all_layers(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """
    返回每层在“最后一个非 padding token”处的隐藏状态。
    形状: [L, N, H] （L=层数(不含或含embedding), N=样本数, H=hidden_size）
    """
    
    device = next(model.parameters()).device
    all_batches = []   # 每个元素: [L, B, H]

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]

        # 注意：需要 tokenizer 有 pad_token（建议设为 eos）
        enc = tokenizer(
            batch,
            return_tensors="pt",
            # padding=True,          # 批内对齐
            padding=False,
            # truncation=True,       # 需要时可改 False，但过长会 OOM/超出位置
            truncation=False,
            add_special_tokens=False,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        print("input_ids.shape:", input_ids.shape)
        print("input_ids:", input_ids)
        print("attention_mask.shape:", attention_mask.shape)
        print("attention_mask:", attention_mask)


        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        # hidden_states 是 tuple: [emb, layer1, layer2, ..., layerN]
        hs_tuple = out.hidden_states

        # 每条样本最后一个非 padding 的位置
        last_idx = attention_mask.sum(dim=1) - 1         # [B]
        b_idx = torch.arange(input_ids.size(0), device=device)

        print(last_idx)
        print(b_idx)


        # hidden_states 是 tuple: [emb, layer1, layer2, ..., layerN]
        per_layer = []
        for h in hs_tuple[1:]:
            # h: [B, T, H] -> 取最后非 pad 位置的向量 [B, H]
            per_layer.append(h[b_idx, last_idx])          # [B, H]

        # [num_layers, B, H]
        batch_stack = torch.stack(per_layer, dim=0).detach().cpu()
        print("batch_stack.shape:", batch_stack.shape)

        all_batches.append(batch_stack)

    # 跨 batch 在维度 1（B 维）拼接 -> [L, N, H]
    all_layers = torch.cat(all_batches, dim=1)


    return all_layers.to(torch.float32).numpy()









# ===================== Prompt 构造（与你原始逻辑保持一致） =====================

def build_prompts(
    tokenizer,
    inputs: List[str],
    contexts: List[str],
    add_assistant_header: bool = True,
    control_length: int = 20,
) -> List[str]:
    """
    仿照你原有的：
      - 用 tokenizer.apply_chat_template 生成 user 消息
      - 添加控制提示（reserved special tokens）
      - 末尾可选拼接 "<|start_header_id|>assistant<|end_header_id|>"
    """


    control_prompt = (
        f"\nSummarize the documents relevant to the question in K sentences, "
        f"where K = <|reserved_special_token_100|>{control_length}<|reserved_special_token_101|>"
    )

    prompts = []
    for q, ctx in zip(inputs, contexts):

        """
        user_text = (
            "Write a high-quality summary of the provided documents with respect to the question.\n"
            f" ### This is the question: {q}\n"
            f"### These are the documents:\n{ctx}\n"
            f"### This is the summary:{control_prompt}"
        )
        """

        
        user_text = (
            "Write a high-quality summary of the provided documents with respect to the question.\n"
            f" ### This is the question: {q}\n"
            f"### These are the documents:\n{ctx}\n"
            f"### This is the summary:"
        )
        
        
        """
        user_text = (
            "### Task Type: Multi-Document Compression\n"
            "Write a high-quality summary of the provided documents with respect to the question.\n"
            f" ### This is the question: {q}\n"
            f"### These are the documents:\n{ctx}\n"
            f"### This is the summary:"
        )
        """

        """
        user_text = (
            "### Task Type: Single-Document Compression\n"
            "Write a high-quality summary of the provided documents with respect to the question.\n"
            f" ### This is the question: {q}\n"
            f"### These are the documents:\n{ctx}\n"
            f"### This is the summary:"
        )
        """

        # 生成“最终送入模型的字符串”
        p = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': user_text}],
            tokenize=False,
        )


        if add_assistant_header:
            p = p + "<|start_header_id|>assistant<|end_header_id|>"
        prompts.append(p)
    
    return prompts


# ===================== 主流程 =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Extract last-token hidden states for prompts (no generation).")

    # 模型 & 设备
    parser.add_argument("--model_path", type=str, required=True, help="Hugging Face 可加载的模型路径或名称")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"], help="模型精度")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu' 或 'auto'")

    # 数据与输出
    parser.add_argument("--dataset_root", type=str,
                        default="/data2/junyizhang/BRIEF_train_eval_Code/LongBench/LongBench/LongBench",
                        help="LongBench 根目录（包含 unbiasdata/ 与 compdata/）")
    parser.add_argument("--out_dir", type=str, default="./hs_out", help="隐藏态输出目录")
    parser.add_argument("--batch_size", type=int, default=8, help="前向批大小")

    # Prompt 构造选项
    parser.add_argument("--no_assistant_header", action="store_true",
                        help="不在 prompt 末尾拼接 '<|start_header_id|>assistant<|end_header_id|>'")
    
    parser.add_argument("--control_length", type=int, default=20, help="K 的值（控制提示中的句子数）")

    # 数据集列表（与原脚本一致）
    parser.add_argument("--datasets", nargs="+",
                        default=["hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en"],
                        help="要处理的数据集名称列表（对应 unbiasdata/{name}.jsonl）")

    return parser.parse_args()


def pick_dtype(device_str: str, want: str):
    """
    if want != "auto":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[want]
    # auto
    if device_str == "cpu":
        return torch.float32
    if torch.cuda.is_available():
        # 优先 bfloat16（A100/H100 等）
        bf16_ok = False
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            # 经验判断：算力 >= 8.0（Ampere）大概率 OK
            major, _ = torch.cuda.get_device_capability(0)
            bf16_ok = major >= 8
        return torch.bfloat16 if bf16_ok else torch.float16
    """

    # return torch.float32
    return torch.bfloat16


def main():
    args = parse_args()

    dataset_root = args.dataset_root

    # unbias_dir = os.path.join(dataset_root, "unbiasdata")
    unbias_dir = os.path.join(dataset_root, "data")

    ensure_dir(args.out_dir)

    print(unbias_dir)


    # 设备与精度
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = pick_dtype(device, args.dtype)

    print(f"[Info] device={device}, dtype={dtype}")
    print(args.model_path)

    # 加载 tokenizer / 模型（一次即可）
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        # low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,  # 多卡自动切分；CPU 时加载到 CPU
        # 如需更省显存，可打开 8bit（需 pip install bitsandbytes）
        # load_in_8bit=True,
    ).eval()

    # print(model)

    add_assistant_header = not args.no_assistant_header


    print(args.datasets)
    # raise ValueError("Here!")


    for dataset in args.datasets:
        in_path = os.path.join(unbias_dir, f"{dataset}.jsonl")
        if not os.path.exists(in_path):
            print(f"[Warn] {in_path} 不存在，跳过该数据集")
            continue

        print(f"\n===== 处理数据集: {dataset} =====")
        data = read_jsonl(in_path)

        # 你原始字段：context -> 文档，input -> 问题
        contexts = [it["context"] for it in data]
        inputs = [it["input"] for it in data]

        # 构造 prompts
        prompts = build_prompts(
            tokenizer=tokenizer,
            inputs=inputs,
            contexts=contexts,
            add_assistant_header=add_assistant_header,
            control_length=args.control_length,
        )

        # 预览
        print("################################  Prompt 示例  ################################")
        # print(prompts[0][:800] + ("..." if len(prompts[0]) > 800 else ""))
        print(prompts[0])
        print("##############################################################################")

        # 取最后 token 隐藏态
        # hs = last_token_hidden_states(model, tokenizer, prompts, batch_size=args.batch_size)  # [N, H]
        hs = last_token_hidden_states_all_layers(model, tokenizer, prompts, batch_size=args.batch_size)  # [N, H]
        print(f"[Info] hidden states shape: {hs.shape}")

        # 保存 .npy
        out_path = os.path.join(args.out_dir, f"{dataset}_last_token_hidden.npy")
        np.save(out_path, hs)
        print(f"[OK] 已保存：{out_path}")

    print("\n=== 全部完成 ===")


if __name__ == "__main__":
    main()




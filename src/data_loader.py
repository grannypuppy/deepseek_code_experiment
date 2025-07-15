# src/data_loader.py
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# --- 数据加载 ---
def load_data_from_jsonl(file_path: str) -> Dataset:
    """从jsonl文件中加载数据并返回Hugging Face Dataset对象"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 检查行是否为空，避免json解析错误
            if line.strip():
                data.append(json.loads(line))
    return Dataset.from_list(data)

# --- SFT (监督微调) 预处理 ---
def preprocess_for_sft(examples, tokenizer: AutoTokenizer):
    """
    为监督微调（SFT）任务预处理数据。
    格式: <BOS> src_code <EOS> <BOS> tgt_code <EOS>
    目标: 模型只预测 tgt_code 部分。
    """
    inputs = []
    labels = []

    # 确保'src_code'和'tgt_code'存在
    if 'src_code' not in examples or 'tgt_code' not in examples:
        return {"input_ids": [], "labels": []}

    for src, tgt in zip(examples['src_code'], examples['tgt_code']):
        src_tokenized = tokenizer(src, truncation=True, max_length=512)
        tgt_tokenized = tokenizer(tgt, truncation=True, max_length=512)

        input_ids = src_tokenized['input_ids'] + tgt_tokenized['input_ids']
        
        src_labels = [-100] * len(src_tokenized['input_ids'])
        tgt_labels = tgt_tokenized['input_ids']
        
        label_ids = src_labels + tgt_labels
        
        if len(input_ids) > tokenizer.model_max_length:
            input_ids = input_ids[:tokenizer.model_max_length]
            label_ids = label_ids[:tokenizer.model_max_length]

        inputs.append(input_ids)
        labels.append(label_ids)

    return {"input_ids": inputs, "labels": labels}

# --- MLM (掩码语言模型) 预处理 ---
def preprocess_for_mlm(examples):
    """
    为掩码语言模型（MLM）任务预处理数据。
    我们只将src_code和tgt_code拼接成一个大的文本语料库。
    真正的“掩码”操作将由Trainer的DataCollator完成。
    """
    # 确保'src_code'和'tgt_code'存在
    if 'src_code' not in examples or 'tgt_code' not in examples:
        return {"text": []}
        
    texts = [src + "\n" + tgt for src, tgt in zip(examples['src_code'], examples['tgt_code'])]
    return {"text": texts}
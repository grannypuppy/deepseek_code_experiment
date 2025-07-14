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

    for src, tgt in zip(examples['src_code'], examples['tgt_code']):
        # 1. 对源和目标代码分别进行编码，并添加特殊token
        src_tokenized = tokenizer(src, truncation=True, max_length=512)
        tgt_tokenized = tokenizer(tgt, truncation=True, max_length=512)

        # 2. 拼接源和目标的token ID
        input_ids = src_tokenized['input_ids'] + tgt_tokenized['input_ids']
        
        # 3. 创建标签，将源语言部分的标签设为-100，这样损失函数会忽略它们
        src_labels = [-100] * len(src_tokenized['input_ids'])
        tgt_labels = tgt_tokenized['input_ids']
        
        # 4. 拼接标签
        label_ids = src_labels + tgt_labels
        
        # 5. 再次进行截断，确保总长度不超过模型限制
        if len(input_ids) > tokenizer.model_max_length:
            input_ids = input_ids[:tokenizer.model_max_length]
            label_ids = label_ids[:tokenizer.model_max_length]

        inputs.append(input_ids)
        labels.append(label_ids)

    # 返回结果，注意需要padding
    # Trainer会自动处理padding，我们只需要返回token id列表
    return {"input_ids": inputs, "labels": labels}

# --- MLM (掩码语言模型) 预处理 ---
def preprocess_for_mlm(examples):
    """
    为掩码语言模型（MLM）任务预处理数据。
    我们只将src_code和tgt_code拼接成一个大的文本语料库。
    真正的“掩码”操作将由Trainer的DataCollator完成。
    """
    # 将源和目标代码合并成一个文本流
    # 使用换行符分隔，以保留代码的结构感
    texts = [src + "\n" + tgt for src, tgt in zip(examples['src_code'], examples['tgt_code'])]
    return {"text": texts}
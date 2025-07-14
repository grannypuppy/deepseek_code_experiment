# src/config.py
import torch
from dataclasses import dataclass, field
from transformers import TrainingArguments
from peft import LoraConfig

# --- 模型和数据路径配置 ---
@dataclass
class PathArguments:
    base_model_path: str = field(default="deepseek-ai/deepseek-coder-1.3b-base", metadata={"help": "基础模型ID"})
    train_file: str = field(default="../data/train_hq_only.txt", metadata={"help": "训练数据文件路径"})
    val_file: str = field(default="../data/val.txt", metadata={"help": "验证数据文件路径"})
    test_file: str = field(default="../data/test.txt", metadata={"help": "测试数据文件路径"})
    # 实验一（任务导向）的模型输出路径
    output_dir_task: str = field(default="../results/task_finetune", metadata={"help": "任务导向型训练的模型保存路径"})
    # 实验二（表征导向）的模型输出路径
    output_dir_rep_stage1: str = field(default="../results/representation_finetune/stage1_mlm", metadata={"help": "表征预训练（MLM）模型保存路径"})
    output_dir_rep_stage2: str = field(default="../results/representation_finetune/stage2_sft", metadata={"help": "表征微调（SFT）模型保存路径"})

# --- 训练总预算和分配 ---
# 假设总训练预算为 N 步，这里我们设定一个具体值，例如 10000
# 您可以根据您的计算资源和时间进行调整
TOTAL_TRAINING_STEPS = 10000

# 实验一：任务导向型训练
# 直接进行SFT，使用全部训练步数
TASK_SFT_STEPS = TOTAL_TRAINING_STEPS

# 实验二：表征导向型训练
# 按照要求，P 和 F 各占一半
REP_PRETRAIN_STEPS_P = TOTAL_TRAINING_STEPS // 2  # P = N / 2
REP_FINETUNE_STEPS_F = TOTAL_TRAINING_STEPS // 2  # F = N / 2

# --- PEFT LoRA 配置 ---
# LoRA (Low-Rank Adaptation) 配置，用于高效微调
lora_config = LoraConfig(
    r=8,  # LoRA的秩，可以设为8, 16, 32等
    lora_alpha=16,  # LoRA的alpha值，通常是r的两倍
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 针对DeepSeek Coder模型的注意力层和MLP层
    lora_dropout=0.05,  # Dropout率
    bias="none",
    task_type="CAUSAL_LM"  # 任务类型为因果语言模型
)

# --- 通用训练参数 ---
# 这里我们定义一些会被两个实验共用的训练参数
def get_base_training_args(output_dir: str, max_steps: int) -> TrainingArguments:
    """获取基础的训练参数配置"""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # 根据您的显存调整
        gradient_accumulation_steps=4,   # 梯度累积，有效batch size = 4 * 4 = 16
        learning_rate=1e-4,              # LoRA微调常用的学习率
        logging_steps=50,                # 每50步记录一次日志
        save_steps=500,                  # 每500步保存一次checkpoint
        max_steps=max_steps,             # 训练总步数
        save_total_limit=2,              # 最多保存2个checkpoint
        fp16=True,                       # 使用FP16混合精度训练以加速
        # bf16=True,                     # 如果您的GPU支持BF16（如A100），建议使用此项
        optim="paged_adamw_8bit",        # 使用8-bit优化器以节省显存
        warmup_steps=100,                # 预热步数
        lr_scheduler_type="cosine",      # 使用余弦学习率衰减
        report_to="tensorboard",         # 将日志报告给Tensorboard
        dataloader_num_workers=4,        # 数据加载进程数
    )

# --- 设备配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
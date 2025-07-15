# src/config.py
import os
import torch
from dataclasses import dataclass, field
from transformers import TrainingArguments
from peft import LoraConfig

# --- W&B 配置 ---
# 设置您的W&B项目名称，所有实验都会记录到这个项目下
WANDB_PROJECT_NAME = "deepseek-coder-comparison"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

# --- 模型和数据路径配置 ---
@dataclass
class PathArguments:
    base_model_path: str = field(default="deepseek-ai/deepseek-coder-1.3b-base", metadata={"help": "基础模型ID"})
    train_file: str = field(default="/research/jiamin0630/deepseek_coder_experiment/data/train_hq_only.jsonl", metadata={"help": "训练数据文件路径"})
    val_file: str = field(default="/research/jiamin0630/deepseek_coder_experiment/data/val.jsonl", metadata={"help": "验证数据文件路径"})
    test_file: str = field(default="/research/jiamin0630/deepseek_coder_experiment/data/test.jsonl", metadata={"help": "测试数据文件路径"})
    output_dir_task: str = field(default="/research/jiamin0630/deepseek_coder_experiment/results/task_finetune", metadata={"help": "任务导向型训练的模型保存路径"})
    output_dir_rep_stage1: str = field(default="/research/jiamin0630/deepseek_coder_experiment/results/representation_finetune/stage1_mlm", metadata={"help": "表征预训练（MLM）模型保存路径"})
    output_dir_rep_stage2: str = field(default="/research/jiamin0630/deepseek_coder_experiment/results/representation_finetune/stage2_sft", metadata={"help": "表征微调（SFT）模型保存路径"})


# --- 训练总预算和分配 ---
TOTAL_TRAINING_STEPS = 10000
TASK_SFT_STEPS = TOTAL_TRAINING_STEPS
REP_PRETRAIN_STEPS_P = TOTAL_TRAINING_STEPS // 2
REP_FINETUNE_STEPS_F = TOTAL_TRAINING_STEPS // 2

# --- PEFT LoRA 配置 ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- 通用训练参数 (核心修改点) ---
def get_base_training_args(output_dir: str, max_steps: int) -> TrainingArguments:
    """获取基础的训练参数配置"""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,  # 我们可以更频繁地记录，以便在wandb上看到更平滑的曲线
        save_steps=500,
        max_steps=max_steps,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        # 【核心修改】: 启用wandb作为报告后端
        # Hugging Face Trainer与wandb深度集成，只需将'wandb'添加到report_to列表中
        # 它会自动记录loss、学习率、评估指标等所有信息。
        report_to="wandb",
        run_name=f"{os.path.basename(output_dir)}-{WANDB_PROJECT_NAME}", # 为每个运行设置一个清晰的名称
        dataloader_num_workers=4,
    )

# --- 设备配置 ---
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# src/train_task.py
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import get_peft_model
from config import (
    PathArguments,
    lora_config,
    get_base_training_args,
    TASK_SFT_STEPS,
    DEVICE
)
from data_loader import load_data_from_jsonl, preprocess_for_sft

def main():

    print(f"当前设备: {DEVICE}")

    # --- 1. 加载配置 ---
    print("实验一：任务导向型训练 (SFT only)")
    parser = HfArgumentParser((PathArguments,))
    path_args, = parser.parse_args_into_dataclasses()

    # get_base_training_args已经配置好了wandb报告
    training_args = get_base_training_args(
        output_dir=path_args.output_dir_task,
        max_steps=TASK_SFT_STEPS
    )

    # --- 2. 加载Tokenizer和模型 ---
    tokenizer = AutoTokenizer.from_pretrained(path_args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )

    # --- 3. PEFT LoRA 配置 ---
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. 加载和预处理数据 ---
    train_dataset = load_data_from_jsonl(path_args.train_file)
    processed_train_dataset = train_dataset.map(
        lambda examples: preprocess_for_sft(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # --- 5. 配置Trainer ---
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        data_collator=data_collator,
    )
    
    # --- 6. 开始训练 ---
    print("开始SFT训练... 训练日志将自动同步到 W&B.")
    trainer.train()

    # --- 7. 保存最终的适配器 ---
    print(f"训练完成，保存LoRA适配器到 '{path_args.output_dir_task}'")
    trainer.save_model(path_args.output_dir_task)
    
    # --- 8. 结束wandb运行 ---
    wandb.finish()


if __name__ == "__main__":
    main()
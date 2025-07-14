# src/train_task.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
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
    # --- 1. 加载配置 ---
    print("实验一：任务导向型训练 (SFT only)")
    parser = HfArgumentParser((PathArguments,))
    path_args, = parser.parse_args_into_dataclasses()

    training_args = get_base_training_args(
        output_dir=path_args.output_dir_task,
        max_steps=TASK_SFT_STEPS
    )

    # --- 2. 加载Tokenizer和模型 ---
    print(f"从 '{path_args.base_model_path}' 加载模型和Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path_args.base_model_path, trust_remote_code=True)
    # 设置padding token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path,
        torch_dtype=torch.bfloat16, # 使用bfloat16以提高效率
        device_map=DEVICE,
        trust_remote_code=True
    )

    # --- 3. PEFT LoRA 配置 ---
    print("应用PEFT LoRA适配器...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. 加载和预处理数据 ---
    print("加载并预处理数据用于SFT...")
    train_dataset = load_data_from_jsonl(path_args.train_file)
    
    # 对数据集进行SFT预处理
    processed_train_dataset = train_dataset.map(
        lambda examples: preprocess_for_sft(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    print(f"处理后得到的训练样本数量: {len(processed_train_dataset)}")
    
    # --- 5. 配置Trainer ---
    # SFT使用DataCollatorForSeq2Seq，它会自动处理padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        data_collator=data_collator,
    )
    
    # --- 6. 开始训练 ---
    print("开始SFT训练...")
    trainer.train()

    # --- 7. 保存最终的适配器 ---
    print(f"训练完成，保存LoRA适配器到 '{path_args.output_dir_task}'")
    trainer.save_model(path_args.output_dir_task)


if __name__ == "__main__":
    main()
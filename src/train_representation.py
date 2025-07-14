# src/train_representation.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer
)
from peft import get_peft_model, PeftModel
from config import (
    PathArguments,
    lora_config,
    get_base_training_args,
    REP_PRETRAIN_STEPS_P,
    REP_FINETUNE_STEPS_F,
    DEVICE
)
from data_loader import load_data_from_jsonl, preprocess_for_mlm, preprocess_for_sft

def main():
    # --- 0. 加载配置 ---
    print("实验二：表征导向型训练 (MLM + SFT)")
    parser = HfArgumentParser((PathArguments,))
    path_args, = parser.parse_args_into_dataclasses()

    # ===================================================================
    #  第一阶段：表征预训练 (Representation Pre-training via MLM)
    # ===================================================================
    print("\n--- [阶段一] 开始表征预训练 (MLM) ---")

    # --- 1.1. 加载模型和Tokenizer (从头开始) ---
    tokenizer = AutoTokenizer.from_pretrained(path_args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model = get_peft_model(model, lora_config)
    print("MLM阶段可训练参数:")
    model.print_trainable_parameters()
    
    # --- 1.2. 加载和预处理数据 (用于MLM) ---
    train_dataset_mlm = load_data_from_jsonl(path_args.train_file)
    processed_train_dataset_mlm = train_dataset_mlm.map(
        preprocess_for_mlm,
        batched=True,
        remove_columns=train_dataset_mlm.column_names
    ).map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        remove_columns=["text"]
    )
    print(f"MLM阶段训练样本数: {len(processed_train_dataset_mlm)}")

    # --- 1.3. 配置MLM的Data Collator ---
    # 这个collator会自动进行token掩码
    # mlm_probability设为0.3（30%），根据你的要求
    data_collator_mlm = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.3)
    
    # --- 1.4. 配置训练参数并开始训练 ---
    training_args_mlm = get_base_training_args(
        output_dir=path_args.output_dir_rep_stage1,
        max_steps=REP_PRETRAIN_STEPS_P
    )
    
    trainer_mlm = Trainer(
        model=model,
        args=training_args_mlm,
        train_dataset=processed_train_dataset_mlm,
        data_collator=data_collator_mlm,
    )
    
    print("开始MLM训练...")
    trainer_mlm.train()
    print(f"MLM训练完成，保存适配器到 '{path_args.output_dir_rep_stage1}'")
    trainer_mlm.save_model(path_args.output_dir_rep_stage1)
    
    # 清理内存
    del model
    del trainer_mlm
    torch.cuda.empty_cache()

    # ===================================================================
    #  第二阶段：监督微调 (Supervised Fine-tuning)
    # ===================================================================
    print("\n--- [阶段二] 开始监督微调 (SFT) ---")

    # --- 2.1. 加载模型，并应用第一阶段训练好的适配器 ---
    print(f"加载基础模型 '{path_args.base_model_path}'...")
    model_sft = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    
    print(f"加载并合并第一阶段训练的MLM适配器从 '{path_args.output_dir_rep_stage1}'...")
    # 加载第一阶段的LoRA权重
    model_sft = PeftModel.from_pretrained(model_sft, path_args.output_dir_rep_stage1)
    # !! 重要：将LoRA权重合并到基础模型中，这样第二阶段的SFT就有了一个更好的起点
    model_sft = model_sft.merge_and_unload()
    # 为第二阶段SFT应用新的LoRA配置
    model_sft = get_peft_model(model_sft, lora_config)

    print("SFT阶段可训练参数:")
    model_sft.print_trainable_parameters()
    
    # --- 2.2. 加载和预处理数据 (用于SFT) ---
    train_dataset_sft = load_data_from_jsonl(path_args.train_file)
    processed_train_dataset_sft = train_dataset_sft.map(
        lambda examples: preprocess_for_sft(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset_sft.column_names
    )
    print(f"SFT阶段训练样本数: {len(processed_train_dataset_sft)}")

    # --- 2.3. 配置SFT的Data Collator ---
    data_collator_sft = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    # --- 2.4. 配置训练参数并开始训练 ---
    training_args_sft = get_base_training_args(
        output_dir=path_args.output_dir_rep_stage2,
        max_steps=REP_FINETUNE_STEPS_F
    )

    trainer_sft = Trainer(
        model=model_sft,
        args=training_args_sft,
        train_dataset=processed_train_dataset_sft,
        data_collator=data_collator_sft,
    )
    
    print("开始SFT训练...")
    trainer_sft.train()
    print(f"SFT训练完成，保存最终适配器到 '{path_args.output_dir_rep_stage2}'")
    trainer_sft.save_model(path_args.output_dir_rep_stage2)

if __name__ == "__main__":
    main()
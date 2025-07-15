# src/train_representation.py
import torch
import wandb
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
    parser = HfArgumentParser((PathArguments,))
    path_args, = parser.parse_args_into_dataclasses()
    
    # ===================================================================
    #  第一阶段：表征预训练 (MLM)
    # ===================================================================
    print("\n--- [阶段一] 开始表征预训练 (MLM) ---")
    
    tokenizer = AutoTokenizer.from_pretrained(path_args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model = get_peft_model(model, lora_config)
    
    train_dataset_mlm = load_data_from_jsonl(path_args.train_file)
    processed_train_dataset_mlm = train_dataset_mlm.map(
        preprocess_for_mlm, batched=True, remove_columns=train_dataset_mlm.column_names
    ).map(
        lambda examples: tokenizer(examples["text"]), batched=True, remove_columns=["text"]
    )
    
    data_collator_mlm = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.3)
    
    training_args_mlm = get_base_training_args(
        output_dir=path_args.output_dir_rep_stage1,
        max_steps=REP_PRETRAIN_STEPS_P
    )
    
    trainer_mlm = Trainer(
        model=model, args=training_args_mlm, train_dataset=processed_train_dataset_mlm, data_collator=data_collator_mlm
    )
    
    print("开始MLM训练... 日志将自动同步到 W&B.")
    trainer_mlm.train()
    trainer_mlm.save_model(path_args.output_dir_rep_stage1)
    wandb.finish()
    
    del model, trainer_mlm
    torch.cuda.empty_cache()

    # ===================================================================
    #  第二阶段：监督微调 (SFT)
    # ===================================================================
    print("\n--- [阶段二] 开始监督微调 (SFT) ---")
    
    model_sft = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model_sft = PeftModel.from_pretrained(model_sft, path_args.output_dir_rep_stage1)
    model_sft = model_sft.merge_and_unload()
    model_sft = get_peft_model(model_sft, lora_config)

    train_dataset_sft = load_data_from_jsonl(path_args.train_file)
    processed_train_dataset_sft = train_dataset_sft.map(
        lambda examples: preprocess_for_sft(examples, tokenizer), batched=True, remove_columns=train_dataset_sft.column_names
    )
    
    data_collator_sft = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    
    training_args_sft = get_base_training_args(
        output_dir=path_args.output_dir_rep_stage2,
        max_steps=REP_FINETUNE_STEPS_F
    )

    trainer_sft = Trainer(
        model=model_sft, args=training_args_sft, train_dataset=processed_train_dataset_sft, data_collator=data_collator_sft
    )
    
    print("开始SFT训练... 日志将自动同步到 W&B.")
    trainer_sft.train()
    trainer_sft.save_model(path_args.output_dir_rep_stage2)
    wandb.finish()

if __name__ == "__main__":
    main()
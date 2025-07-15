# src/evaluate.py
import torch
import json
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft import PeftModel
from config import PathArguments, DEVICE, WANDB_PROJECT_NAME
from data_loader import load_data_from_jsonl
from tqdm import tqdm
import os

def main():
    # --- 1. 解析参数 ---
    parser = HfArgumentParser((PathArguments,))
    parser.add_argument("--adapter_path", type=str, required=True, help="要评估的已训练LoRA适配器的路径")
    parser.add_argument("--run_name", type=str, required=True, help="为W&B运行设置一个名称, e.g., 'eval-task-finetune'")
    path_args, other_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # --- 2. 初始化W&B运行 ---
    # 为评估过程创建一个新的W&B运行，用于记录表格和指标
    wandb.init(project=WANDB_PROJECT_NAME, name=other_args.run_name, job_type="evaluation")

    # --- 3. 加载模型和Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(path_args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    
    # --- 4. 加载LoRA适配器 ---
    model = PeftModel.from_pretrained(model, other_args.adapter_path)
    model.eval()

    # --- 5. 加载测试数据 ---
    test_dataset = load_data_from_jsonl(path_args.test_file)
    
    # --- 6. 创建W&B表格 ---
    # 我们将创建一个表格来可视化模型的输入、输出和标准答案
    # 根据您提供的数据样本，我们加入 'speedup' 列
    results_table = wandb.Table(columns=["problem_id", "speedup", "src_code", "generated_code", "reference_tgt_code"])

    # --- 7. 循环推理并记录结果 ---
    total_speedup = 0
    for example in tqdm(test_dataset, desc="正在评估并记录到W&B"):
        src_code = example.get('src_code', '')
        if not src_code:
            continue
            
        inputs = tokenizer(src_code, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id, do_sample=True, top_p=0.9, temperature=0.7
            )
        
        generated_code = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        problem_id = example.get('problem_id', 'N/A')
        reference_tgt_code = example.get('tgt_code', '')
        
        # [cite_start]从您的数据样本中获取speedup值 [cite: 1, 3, 5, 7]
        speedup = example.get('speedup', 0.0) 
        total_speedup += speedup

        # 将这一行的结果添加到W&B表格中
        results_table.add_data(problem_id, speedup, src_code, generated_code, reference_tgt_code)

    # --- 8. 记录表格和最终指标到W&B ---
    print("评估完成，正在将结果表格上传到W&B...")
    wandb.log({"evaluation_results": results_table})
    
    # 计算并记录平均指标
    avg_speedup = total_speedup / len(test_dataset) if len(test_dataset) > 0 else 0
    wandb.summary["average_speedup"] = avg_speedup
    # 如果您有其他指标（如pass@k），也在这里记录
    # wandb.summary["pass_at_1"] = calculate_pass_at_k(...)

    # --- 9. 结束W&B运行 ---
    wandb.finish()
    print(f"评估结果已成功记录到W&B项目 '{WANDB_PROJECT_NAME}' 的运行 '{other_args.run_name}' 中。")


if __name__ == "__main__":
    main()
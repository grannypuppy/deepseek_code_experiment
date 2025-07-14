# src/evaluate.py
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from peft import PeftModel
from config import PathArguments, DEVICE
from data_loader import load_data_from_jsonl
from tqdm import tqdm

def main():
    # --- 1. 解析参数 ---
    # 我们复用PathArguments，并添加一个自定义参数来指定要评估哪个适配器
    parser = HfArgumentParser((PathArguments,))
    parser.add_argument("--adapter_path", type=str, required=True, help="要评估的已训练LoRA适配器的路径")
    path_args, other_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # --- 2. 加载模型和Tokenizer ---
    print("加载基础模型和Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path_args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path_args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )

    # --- 3. 加载LoRA适配器 ---
    print(f"加载LoRA适配器从: {other_args.adapter_path}")
    model = PeftModel.from_pretrained(model, other_args.adapter_path)
    model.eval() # 设置为评估模式

    # --- 4. 加载测试数据 ---
    print(f"加载测试数据从: {path_args.test_file}")
    test_dataset = load_data_from_jsonl(path_args.test_file)

    # --- 5. 循环推理并打印结果 ---
    results = []
    for example in tqdm(test_dataset, desc="正在评估"):
        src_code = example['src_code']
        # 我们只使用src_code作为输入
        inputs = tokenizer(src_code, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            # 生成代码，可以调整max_new_tokens等参数
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True, # 使用采样策略
                top_p=0.9,
                temperature=0.7
            )
        
        # 解码生成的token，跳过输入的prompt部分
        generated_code = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        print("-" * 20)
        print(f"Problem ID: {example.get('problem_id', 'N/A')}")
        print("原始输入 (src_code):")
        print(src_code)
        print("\n模型输出 (generated_code):")
        print(generated_code)
        print("-" * 20 + "\n")

        # 这里您可以集成您自己的评估脚本逻辑
        # 例如，调用 gen_eval.py 并传入 src_code, generated_code 和 problem_id
        # 此处仅作演示，将结果保存在列表中
        results.append({
            "problem_id": example.get('problem_id', 'N/A'),
            "src_code": src_code,
            "generated_code": generated_code,
            "reference_tgt_code": example.get('tgt_code', '')
        })

    # 将结果保存到文件
    output_eval_file = f"{other_args.adapter_path}/eval_results.json"
    with open(output_eval_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存到: {output_eval_file}")


if __name__ == "__main__":
    main()
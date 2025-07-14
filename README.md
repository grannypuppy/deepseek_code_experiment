# DeepSeek Coder 训练策略对比实验

本项目旨在比较两种不同的训练策略在`deepseek-coder-1.3b-base`模型上的表现：
1.  **任务导向型训练**: 直接对模型进行监督微调（SFT）。
2.  **表征导向型训练**: 先进行掩码语言模型（MLM）的表征预训练，再进行监督微调（SFT）。

## 1. 环境设置

首先，请确保您已安装所需的依赖库。

```bash
pip install -r requirements.txt
```

请将您的数据集文件 `train_hq_only.txt`, `val.txt`, `test.txt` 放入 `data/` 目录下。

## 2. 运行实验

所有配置（如模型路径、学习率、训练步数等）都可以在 `src/config.py` 文件中进行修改。

### 实验一：任务导向型训练

此脚本将直接在 `train_hq_only` 数据集上对模型进行SFT训练。

```bash
cd src
python train_task.py
```

训练完成后，LoRA适配器将保存在 `results/task_finetune/` 目录下。

### 实验二：表征导向型训练

此脚本将首先进行MLM预训练，然后进行SFT微调。

```bash
cd src
python train_representation.py
```

-   第一阶段（MLM）的适配器保存在 `results/representation_finetune/stage1_mlm/`。
-   第二阶段（SFT）的最终适配器保存在 `results/representation_finetune/stage2_sft/`。

## 3. 评估模型

使用 `evaluate.py` 脚本加载训练好的LoRA适配器，并在 `test.txt` 数据集上进行推理。

**评估实验一的模型:**

```bash
cd src
# 注意 --adapter_path 指向你训练好的适配器目录
python evaluate.py --adapter_path ../results/task_finetune
```

**评估实验二的模型:**

```bash
cd src
# 注意 --adapter_path 指向实验二最终产出的适配器目录
python evaluate.py --adapter_path ../results/representation_finetune/stage2_sft
```

评估脚本会在控制台打印出每个测试用例的输入和模型输出，并将所有结果汇总保存在相应适配器目录下的 `eval_results.json` 文件中。您可以在此脚本中进一步集成您自己的 `gen_eval.py` 逻辑以进行自动化评分。
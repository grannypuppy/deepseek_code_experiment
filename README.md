# DeepSeek Coder 训练策略对比实验 (集成W&B)

本项目旨在比较两种不同的训练策略在`deepseek-coder-1.3b-base`模型上的表现，并通过**Weights & Biases (W&B)** 进行全面的实验跟踪和可视化。

## 1. 环境设置

首先，请确保您已安装所需的依赖库。

```bash
pip install -r requirements.txt
```

**重要：设置Weights & Biases**
您需要一个W&B账户。然后登录：

```bash
wandb login
# 按照提示粘贴您的API密钥
```

请将您的数据集文件 `train_hq_only.txt`, `val.txt`, `test.txt` 放入 `data/` 目录下。

## 2. 运行实验

所有配置（包括W&B项目名称）都可以在 `src/config.py` 文件中进行修改。

### 实验一：任务导向型训练

此脚本将直接进行SFT训练，训练过程中的`loss`、`learning_rate`等指标将自动实时同步到W&B。

```bash
cd src
python train_task.py
```

### 实验二：表征导向型训练

此脚本将分两阶段进行训练，每个阶段都会在W&B上创建一个独立的运行（Run），方便对比。

```bash
cd src
python train_representation.py
```

登录您的W&B账户，进入名为`deepseek-coder-comparison`的项目，即可看到实时更新的图表。

## 3. 评估模型并可视化结果

使用 `evaluate.py` 脚本加载训练好的适配器，在测试集上进行推理，并将详细的生成结果上传到W&B的一个交互式表格中。

**评估实验一的模型:**

```bash
cd src
# 为W&B运行指定一个清晰的名称
python evaluate.py --adapter_path ../results/task_finetune --run_name "eval-task-finetune"
```

**评估实验二的模型:**

```bash
cd src
python evaluate.py --adapter_path ../results/representation_finetune/stage2_sft --run_name "eval-representation-finetune"
```

评估完成后，进入对应的W&B运行页面，您将看到一个名为`evaluation_results`的表格。您可以方便地在该表格中筛选、排序和比较每个测试用例的`src_code`、`generated_code`和`reference_tgt_code`，从而直观地评估模型性能。
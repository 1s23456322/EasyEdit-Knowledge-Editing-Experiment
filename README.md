# EasyEdit Knowledge Editing Experiments

本项目基于 EasyEdit 框架完成知识编辑实验，主要包含四个任务：

- **Task 1：基础环境搭建与基线测试 Baseline Evaluation**
- **Task 2：ROME 单条事实编辑 Single Fact Editing**
- **Task 3：MEMIT 批量知识编辑 Batch Editing**
- **Task 4：综合评估 Comprehensive Evaluation**

实验模型主要使用：

```text
gpt2-medium
```

实验框架使用：

```text
EasyEdit
```

EasyEdit 代码版本：

```text
https://github.com/zjunlp/EasyEdit/tree/f5857dd26b0935a0b4b08d5e88d97ba8cc291e2b
```

---

## 1. 实验环境

### 1.1 硬件环境

本实验使用的 GPU 为：

```text
NVIDIA GeForce RTX 3050 Laptop GPU
```

PyTorch CUDA 检查结果：

```text
torch: 2.7.1+cu118
torch.cuda.is_available(): True
CUDA: 11.8
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

### 1.2 软件环境

推荐使用 Conda 创建实验环境：

```bat
conda create -n easyedit python=3.9 -y
conda activate easyedit
```

先安装 CUDA 版 PyTorch：

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

然后安装其他依赖：

```bat
pip install -r requirements.txt
```

---

## 2. requirements.txt

本项目的 `requirements.txt` 内容如下：

```txt
# Core scientific packages
numpy
scipy
pandas
scikit-learn
tqdm
requests
pyyaml

# HuggingFace ecosystem
transformers==4.30.1
huggingface_hub==0.14.1
accelerate==0.21.0
datasets==1.18.3
pyarrow==12.0.1
sentencepiece

# EasyEdit related dependencies
fairscale
higher
nltk
einops
timm
sentence-transformers
peft==0.4.0

# Logging / utility
matplotlib
```

注意：本项目不建议把 `torch` 写入 `requirements.txt`，因为 PyTorch 需要根据本机 CUDA 版本单独安装。否则直接执行 `pip install -r requirements.txt` 可能会覆盖已经安装好的 GPU 版 PyTorch。

---

## 3. EasyEdit 源码准备

克隆 EasyEdit：

```bat
git clone https://github.com/zjunlp/EasyEdit.git
cd EasyEdit
git checkout f5857dd26b0935a0b4b08d5e88d97ba8cc291e2b
```

测试 EasyEdit 是否能正常导入：

```bat
python -c "from easyeditor import BaseEditor, ROMEHyperParams; print('EasyEdit OK')"
```

如果输出：

```text
EasyEdit OK
```

说明 EasyEdit 环境配置成功。

测试 GPU：

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 4. 项目目录结构

建议项目目录如下：

```text
D:\LLM
├── EasyEdit
├── data.json
├── baseline.py
├── run_rome_gpt2_medium.py
├── run_rome_summary.py
├── counterfact_500_easyedit.json
├── prepare_counterfact_500.py
├── run_memit_500.py
├── evaluate.py
├── baseline_results.json
├── rome_gpt2_medium_results.json
├── memit_500_results.json
├── memit_gpu_log.csv
├── evaluation_summary.json
├── requirements.txt
└── README.md
```

---

## 5. 重要兼容性修改

### 5.1 HuggingFace 版本兼容

本实验使用的 EasyEdit commit 较旧，因此需要固定部分 HuggingFace 相关依赖版本：

```text
transformers==4.30.1
huggingface_hub==0.14.1
accelerate==0.21.0
datasets==1.18.3
pyarrow==12.0.1
```

不要随意升级以下包：

```bat
pip install -U transformers
pip install -U huggingface_hub
pip install -U datasets
pip install -U pyarrow
pip install -U accelerate
```

否则可能出现如下错误：

```text
ImportError: cannot import name 'cached_download'
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
ImportError: cannot import name 'split_torch_state_dict_into_shards'
ImportError: cannot import name 'is_npu_available'
```

### 5.2 Windows 下 MEMIT 的 DataLoader 修改

在 Windows 上运行 MEMIT 时，可能出现：

```text
Can't pickle local object 'length_collation.<locals>.collate_fn'
```

需要修改文件：

```text
D:\LLM\EasyEdit\easyeditor\util\runningstats.py
```

找到类似代码：

```python
return torch.utils.data.DataLoader(
    dataset, sampler=sampler, batch_size=batch_size, **kwargs
)
```

在其前面加入：

```python
kwargs["num_workers"] = 0
```

修改后应为：

```python
kwargs["num_workers"] = 0

return torch.utils.data.DataLoader(
    dataset, sampler=sampler, batch_size=batch_size, **kwargs
)
```

这样可以避免 Windows 多进程 pickle 错误。

---

## 6. Task 1：Baseline Evaluation

### 6.1 任务目标

在不进行任何编辑操作的情况下，测试原始 `gpt2-medium` 对 10 条事实型 Prompt 的回答情况。该任务用于证明模型在编辑前存在旧知识、错误知识或目标知识缺失。

### 6.2 数据文件

数据文件为：

```text
D:\LLM\data.json
```

每条数据包含如下字段：

```json
{
  "prompt": "...",
  "subject": "...",
  "ground_truth": "...",
  "target_new": "...",
  "rephrase_prompt": "...",
  "locality_prompt": "...",
  "locality_ground_truth": "..."
}
```

### 6.3 运行命令

```bat
cd /d D:\LLM
python baseline.py
```

运行后生成：

```text
D:\LLM\baseline_results.json
```

### 6.4 Baseline 结果分析

本次 baseline 测试是在不进行任何知识编辑的情况下，直接使用 `gpt2-medium` 对 10 条事实型 Prompt 进行生成测试。结果显示，模型在 10 条样本中只有 2 条命中了 `target_new`，分别是：

```text
The capital of Australia is -> Sydney
The planet closest to the Sun is -> Earth
```

其余 8 条样本没有输出目标新答案，说明模型在编辑前并不能稳定生成希望注入的新事实。

| 统计项 | 数量 | 比例 |
|---|---:|---:|
| 命中目标新答案 target_new | 2 / 10 | 20% |
| 未命中目标新答案 target_new | 8 / 10 | 80% |
| 命中原始旧答案 ground_truth | 4 / 10 | 40% |
| 未命中原始旧答案 ground_truth | 6 / 10 | 60% |

从具体样本看，模型存在三类情况：

1. **输出旧知识或原事实**：例如日本货币问题输出 `yen`，水的化学式问题输出 `H2O`。
2. **输出错误或无关内容**：例如输入 `The current CEO of Twitter is` 时，模型输出 `Jack Dorsey`。
3. **编辑前已经命中目标答案**：例如澳大利亚首都和太阳最近行星两条样本在 baseline 阶段已经命中目标答案。

总体来看，baseline 结果说明 `gpt2-medium` 在这些事实型问题上的回答并不稳定，存在旧知识、错误知识和目标知识缺失等情况，因此后续使用 ROME 和 MEMIT 进行知识编辑是有必要的。

---

## 7. Task 2：ROME Single Fact Editing

### 7.1 任务目标

使用 ROME 方法对模型进行单条事实编辑。实验中对 `data.json` 中的 10 条事实逐条进行编辑，每次编辑前重新加载原始模型，保证各样本之间互不影响。

### 7.2 ROME 参数文件

ROME 参数文件为：

```text
D:\LLM\EasyEdit\hparams\ROME\gpt2-medium.yaml
```

核心配置如下：

```yaml
alg_name: "ROME"
model_name: "gpt2-medium"

stats_dir: "./data/stats"
device: 0

layers: [8]
fact_token: "subject_last"

v_num_grad_steps: 20
v_lr: 5e-1
v_loss_layer: 23
v_weight_decay: 0.5
clamp_norm_factor: 4
kl_factor: 0.0625

mom2_adjustment: false
context_template_length_params: [[5, 10], [10, 10]]

rewrite_module_tmp: "transformer.h.{}.mlp.c_proj"
layer_module_tmp: "transformer.h.{}"
mlp_module_tmp: "transformer.h.{}.mlp"
attn_module_tmp: "transformer.h.{}.attn"

ln_f_module: "transformer.ln_f"
lm_head_module: "lm_head"

mom2_dataset: "wikipedia"
mom2_n_samples: 100000
mom2_dtype: "float32"

model_parallel: false
```

### 7.3 运行完整 ROME 实验

```bat
cd /d D:\LLM
python run_rome_gpt2_medium.py
```

输出结果文件：

```text
D:\LLM\rome_gpt2_medium_results.json
```

### 7.4 运行简洁截图版 ROME 实验

如果只想得到适合截图提交的简洁输出，运行：

```bat
cd /d D:\LLM
python run_rome_summary.py
```

该脚本会屏蔽 ROME 内部的 loss、vector、delta 等详细日志，只输出 10 条样本编辑前后的简洁结果。

### 7.5 ROME 实验结果

最终 ROME 编辑结果为：

| 指标 | 结果 |
|---|---:|
| 原 Prompt 编辑成功率 | 10 / 10 = 100% |
| Rephrase 泛化成功率 | 7 / 10 = 70% |
| Locality 保持率 | 6 / 10 = 60% |

结论：ROME 在原始 Prompt 上的编辑效果较好，能够让模型在直接编辑问题上输出目标答案；但在同义改写和局部性保持方面并不完全稳定。

---

## 8. Task 3：MEMIT Batch Editing

### 8.1 任务目标

使用 MEMIT 算法进行批量知识编辑。原任务要求从开源知识编辑数据集中选取 500 条数据作为批量编辑集。由于本地实验设备为 RTX 3050 Laptop GPU，显存和磁盘资源有限，因此本实验实际完成的是 CounterFact 数据集中 50 条样本的 MEMIT 批量编辑，并选取前 20 条样本进行验证。

### 8.2 MEMIT 参数文件

MEMIT 参数文件为：

```text
D:\LLM\EasyEdit\hparams\MEMIT\gpt2-medium.yaml
```

核心配置如下：

```yaml
alg_name: "MEMIT"
model_name: "gpt2-medium"
stats_dir: "./data/stats"
device: 0

layers: [4, 5, 6, 7, 8]
layer_selection: "all"
fact_token: "subject_last"

v_num_grad_steps: 20
v_lr: 5e-1
v_loss_layer: 23
v_weight_decay: 0.5
clamp_norm_factor: 0.75
kl_factor: 0.0625

mom2_adjustment: false
mom2_update_weight: 20000
mom2_dataset: "wikipedia"
mom2_n_samples: 1000
mom2_dtype: "float32"

rewrite_module_tmp: "transformer.h.{}.mlp.c_proj"
layer_module_tmp: "transformer.h.{}"
mlp_module_tmp: "transformer.h.{}.mlp"
attn_module_tmp: "transformer.h.{}.attn"

ln_f_module: "transformer.ln_f"
lm_head_module: "transformer.wte"

model_parallel: false
```

### 8.3 准备 CounterFact 数据

运行：

```bat
cd /d D:\LLM
python prepare_counterfact_500.py
```

生成：

```text
D:\LLM\counterfact_500_easyedit.json
```

该文件从 CounterFact 中抽取 500 条样本，并转换为 EasyEdit 可用格式。

### 8.4 设置 HuggingFace 数据集缓存

MEMIT 会使用 Wikipedia 统计数据。为避免 C 盘空间不足，建议把数据集缓存切换到 D 盘：

```bat
cd /d D:\LLM
set HF_DATASETS_CACHE=D:\LLM\hf_cache\datasets
```

注意：该命令只对当前 CMD 窗口有效。运行 MEMIT 时不要关闭当前 CMD。

### 8.5 运行 MEMIT 批量编辑

运行：

```bat
cd /d D:\LLM
python run_memit_500.py
```

本实验中实际设置：

```python
NUM_EDITS = 50
EVAL_COUNT = 20
```

含义如下：

```text
NUM_EDITS = 50   表示一次性批量编辑 50 条知识
EVAL_COUNT = 20  表示编辑后选取前 20 条样本计算 ES / PS / NS
```

运行后生成：

```text
D:\LLM\memit_500_results.json
D:\LLM\memit_gpu_log.csv
```

其中：

- `memit_500_results.json` 保存 MEMIT 编辑结果、耗时、显存和评估数据。
- `memit_gpu_log.csv` 保存运行过程中每个时间点的 GPU 显存占用。

### 8.6 MEMIT 运行结果

本次 MEMIT 运行结果如下：

| 项目 | 结果 |
|---|---:|
| 算法 | MEMIT |
| 模型 | gpt2-medium |
| 批量编辑数量 | 50 |
| 验证样本数量 | 20 |
| 编辑耗时 | 333.64 s |
| 峰值显存占用 nvidia-smi | 3785 MB |
| PyTorch peak allocated | 1954.39 MB |
| PyTorch peak reserved | 2246.0 MB |
| 运行状态 | 成功 |
| 错误信息 | None |

---

## 9. Task 4：Comprehensive Evaluation

### 9.1 任务目标

Task 4 计算以下三个核心指标：

| 指标 | 含义 |
|---|---|
| ES / Efficacy | 直接编辑 Prompt 是否输出目标答案 |
| PS / Generalization | 同义改写 Prompt 是否输出目标答案 |
| NS / Locality | 无关事实回答是否保持不变 |

### 9.2 运行评估脚本

运行：

```bat
cd /d D:\LLM
python evaluate.py
```

输入文件：

```text
D:\LLM\memit_500_results.json
```

输出文件：

```text
D:\LLM\evaluation_summary.json
```

### 9.3 Task 4 评估结果

| 指标 | 含义 | 成功数 / 验证数 | 百分比 |
|---|---|---:|---:|
| ES / Efficacy | 直接编辑 Prompt 是否输出目标答案 | 16 / 20 | 80.00% |
| PS / Generalization | 同义改写 Prompt 是否输出目标答案 | 11 / 20 | 55.00% |
| NS / Locality | 无关事实回答是否保持不变 | 20 / 20 | 100.00% |

### 9.4 Task 4 结果分析

从结果可以看出，MEMIT 在直接编辑 Prompt 上取得了较好的效果，ES 达到 80.00%，说明大多数直接编辑问题能够输出目标答案。PS 为 55.00%，说明模型在同义改写 Prompt 上的泛化能力较弱，很多事实虽然在原始 Prompt 上编辑成功，但换一种问法后仍然不能稳定输出目标答案。NS 为 100.00%，说明在本次验证的 20 条样本中，无关事实回答没有明显被破坏。

需要说明的是，本实验使用的 CounterFact 转换数据中没有 `locality_ground_truth` 字段，因此 NS 使用的是“编辑前后 locality prompt 输出是否保持相近”的近似计算方式，而不是严格判断无关事实是否等于标准答案。因此，该结果只能说明本次验证样本中局部扰动较小，不能完全代表模型所有无关知识均保持正确。

---

## 10. MEMIT 批量编辑结果分析与失败案例总结

### 10.1 总体分析

本实验使用 MEMIT 算法对 `gpt2-medium` 进行批量知识编辑。实验从 CounterFact 数据集中选取 50 条样本作为批量编辑集，并在编辑完成后选取前 20 条样本进行综合评估。最终结果显示：

```text
ES = 80.00%
PS = 55.00%
NS = 100.00%
```

这说明 MEMIT 能够较有效地完成批量知识注入，在直接编辑 Prompt 上表现较好。但是，PS 只有 55.00%，说明 MEMIT 的编辑效果并不能完全泛化到同义改写 Prompt。模型在原始 Prompt 上可能已经学会输出目标答案，但当问题换一种表达方式时，模型仍然可能回到旧知识、输出无关内容，或者受到改写句子中上下文噪声的干扰。

NS 达到 100.00%，说明在本次验证的 20 条样本中，模型对无关事实的回答没有被明显破坏。但由于本实验采用的是编辑前后输出一致性的近似 NS 计算方式，因此该指标只能说明局部扰动较小，不能完全证明模型所有无关知识均保持正确。

### 10.2 失败案例 1：直接编辑失败

样本 2 的目标是：

```text
Prompt: What is the twin city of Shanghai? It is
Target new: Dresden
```

编辑后模型输出：

```text
the capital of China, and the largest city in
```

没有输出目标答案 `Dresden`，因此 ES 失败。这说明对于某些关系表达较复杂或模型本身理解不稳定的 Prompt，MEMIT 即使完成了批量编辑，也不一定能让模型在直接 Prompt 上输出目标对象。

### 10.3 失败案例 2：原始 Prompt 仍然回到旧知识

样本 3 的目标是将：

```text
2011 Cannes Film Festival can be found in
```

编辑为：

```text
Prescott
```

但编辑后模型输出：

```text
the beautiful French Riviera town of Cannes
```

这里模型仍然输出了与旧知识相关的 `Cannes`，没有命中目标答案 `Prescott`。这类失败说明，当模型对某些实体和事实具有较强的原始记忆时，MEMIT 的编辑可能不足以覆盖模型原有知识。

### 10.4 失败案例 3：同义改写泛化失败

样本 1 在直接 Prompt 上编辑成功：

```text
Angola is located in -> Antarctica
```

编辑后模型确实输出了 `Antarctica`。但是在同义改写 Prompt：

```text
Plaque - located at Punkeydoodle's Corners. Angola belongs to the continent of
```

模型输出：

```text
Africa
```

没有输出目标答案 `Antarctica`，因此 PS 失败。这个例子说明，虽然 MEMIT 能够让模型在直接编辑 Prompt 上输出目标答案，但当输入形式变复杂、带有额外上下文噪声时，编辑效果可能无法泛化。

### 10.5 失败案例 4：同义改写受到无关上下文干扰

样本 6 的目标是将 Lady Gaga 的相关答案编辑为：

```text
violin
```

直接 Prompt 上模型输出：

```text
violinist
```

因此 ES 成功。但在 rephrase prompt 中，模型输出：

```text
role of the mother of the child
```

没有命中 `violin`，导致 PS 失败。该样本说明，CounterFact 的 rephrase prompt 中往往包含额外的上下文片段，这些上下文可能会干扰模型生成，使模型没有继续围绕目标事实作答。

### 10.6 失败案例 5：长实体或复杂实体编辑失败

样本 20 的 subject 是：

```text
Elizabeth Boleyn, Countess of Wiltshire
```

目标答案是：

```text
Egypt
```

但编辑后模型输出：

```text
the village of Wiltshire, England
```

没有输出 `Egypt`，ES 失败。这个失败可能与 subject 较长、实体边界复杂有关。MEMIT 和 ROME 类方法都需要定位 subject 表示，长实体或带有头衔、逗号的复杂实体可能导致定位不稳定，从而影响编辑效果。

### 10.7 综合结论

综合来看，MEMIT 适合用于一次性注入多条事实知识，并且在直接 Prompt 上具有较好的编辑效果。但其泛化能力仍然有限，尤其是在 Prompt 改写、长实体、复杂上下文和模型原有知识较强的情况下，仍可能出现编辑失败或泛化失败。

---

## 11. 完整运行顺序

完整运行顺序如下：

```bat
conda activate easyedit

cd /d D:\LLM

python baseline.py
python run_rome_gpt2_medium.py
python run_rome_summary.py

python prepare_counterfact_500.py

set HF_DATASETS_CACHE=D:\LLM\hf_cache\datasets
python run_memit_500.py

python evaluate.py
```

---

## 12. 结果文件说明

| 文件 | 说明 |
|---|---|
| `baseline_results.json` | Task 1 基线测试结果 |
| `rome_gpt2_medium_results.json` | Task 2 ROME 单事实编辑完整结果 |
| `counterfact_500_easyedit.json` | Task 3 使用的 CounterFact 批量编辑数据 |
| `memit_500_results.json` | Task 3 MEMIT 批量编辑结果 |
| `memit_gpu_log.csv` | Task 3 显存占用日志 |
| `evaluation_summary.json` | Task 4 ES / PS / NS 综合评估结果 |

---

## 13. 常见问题

### 13.1 ImportError: cached_download

如果出现：

```text
ImportError: cannot import name 'cached_download'
```

说明 `huggingface_hub` 版本太新，应使用：

```bat
pip install huggingface_hub==0.14.1
```

### 13.2 pyarrow PyExtensionType 错误

如果出现：

```text
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```

应使用：

```bat
pip install pyarrow==12.0.1
```

### 13.3 Windows pickle 错误

如果出现：

```text
Can't pickle local object 'length_collation.<locals>.collate_fn'
```

需要修改：

```text
D:\LLM\EasyEdit\easyeditor\util\runningstats.py
```

将 DataLoader 的 `num_workers` 强制设置为 0。

---

## 14. 实验说明

由于本地设备为 RTX 3050 Laptop GPU，显存和磁盘资源有限，Task 3 中实际完成的是 50 条 CounterFact 样本的 MEMIT 批量编辑，并在前 20 条样本上计算 ES / PS / NS。该设置可以验证 MEMIT 批量知识编辑流程、显存占用、耗时情况以及综合评估指标。

# Qwen2.5-7B-Instruct 微调项目骨架

这是一个适合上手实操的最小可用项目，默认以 `Qwen/Qwen2.5-7B-Instruct` 为底座模型，先走 `LoRA / QLoRA + SFT` 路线。这样做的好处是显存要求更友好，也更适合第一次在服务器上把流程跑通。

## 目录结构

```text
project/
  data/
    train.jsonl
    val.jsonl
  src/
    train.py
    infer.py
  requirements.txt
  README.md
```

## 数据格式

推荐直接使用聊天格式的 JSONL，每行一条样本：

```json
{"messages":[
  {"role":"system","content":"你是一个严谨、清晰的中文助手。"},
  {"role":"user","content":"什么是监督微调（SFT）？"},
  {"role":"assistant","content":"监督微调（SFT）是用带标准答案的样本继续训练预训练模型，让它更符合特定任务、风格或领域要求的过程。"}
]}
```

`train.py` 会调用 tokenizer 的 chat template，把 `messages` 直接拼成训练文本，所以后面你只要持续往 `train.jsonl` 和 `val.jsonl` 填数据即可。

## 建议的实操路线

第一阶段先不要上来就做全参数微调，建议这样走：

1. 先用几十到几百条高质量样本把完整训练流程跑通。
2. 确认训练、保存 adapter、推理加载 adapter 都没问题。
3. 再逐步扩到几千条数据，观察 loss、回复风格和泛化情况。
4. 如果效果稳定，再考虑更长上下文、更大数据量或多卡训练。

## 环境建议

- 本地电脑主要负责整理数据、改脚本、做版本管理。
- 真正训练建议放到 Linux + NVIDIA CUDA 服务器上。
- 第一次实操优先考虑单卡 QLoRA。

一个比较实用的经验值：

- `7B + LoRA`：24GB 显存起步比较稳妥。
- `7B + QLoRA(4bit)`：16GB 到 24GB 也有机会跑起来，但 batch size 和 max length 要保守。
- `7B 全参数微调`：第一次不建议，显存、训练稳定性和工程复杂度都会高很多。

如果你的机器是 `2 x V100-SXM2-16GB`，建议注意这几点：

- 这个配置可以做 `Qwen2.5-7B-Instruct` 的 LoRA / QLoRA 实操。
- V100 更适合 `fp16`，不建议把首轮训练建立在 `bf16` 之上。
- 两张卡在常规 DDP 下主要提升吞吐和有效 batch，不是简单把显存合成 32GB。
- 第一轮建议从 `max_length=1024` 起步，稳定后再尝试 `1536` 或 `2048`。
- 如果只是几十到几百条数据的小实验，单卡先跑通也完全合理。

## 安装依赖

建议在服务器上新建虚拟环境后安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

说明：

- `bitsandbytes` 主要面向 Linux CUDA 环境，macOS 本地一般不适合作为正式训练环境。
- 如果后面上 A100 / H100 / L40S 这类卡，可以优先尝试 `--bf16`。

## 训练示例

在 `project/` 目录下运行：

```bash
python src/train.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train.jsonl \
  --val_file data/val.jsonl \
  --output_dir outputs/qwen2.5-7b-lora \
  --use_4bit \
  --bf16 \
  --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 2048
```

如果你的服务器显卡不是特别新，可以先去掉 `--bf16`，脚本会自动改走 `fp16`。

## 针对 2 x V100 16GB 的推荐启动方式

项目里已经附了一个更保守的启动脚本：

```bash
bash run_train_v100_2x16g.sh
```

这个脚本做了几件事：

- 使用 `torchrun` 以 2 卡 DDP 启动。
- 默认开启 `4bit QLoRA` 和 `gradient checkpointing`。
- 默认把 `max_length` 设为 `1024`，优先保证首轮稳定。
- 不启用 `--bf16`，因为 V100 首选 `fp16`。

如果你第一轮跑通且显存还有余量，可以按这个顺序逐步加：

1. 先把 `max_length` 从 `1024` 提到 `1536`。
2. 再考虑提到 `2048`。
3. 最后再根据 loss 曲线调 `learning_rate` 或 `gradient_accumulation_steps`。

## 推理示例

```bash
python src/infer.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/qwen2.5-7b-lora-v100x2 \
  --prompt "请用通俗的话解释一下 LoRA 为什么省显存" \
  --load_in_4bit
```

## 我给你的几个直接建议

1. 先把数据格式固定下来，不要一边训练一边改 schema。
2. `train` 和 `val` 一开始就分开，哪怕验证集只有几十条也比没有强。
3. 第一轮的目标不是“训得很强”，而是“流程跑通且能稳定复现”。
4. 如果后面要做中文垂类任务，样本质量通常比盲目堆数量更重要。
5. 建议把每次实验的关键参数记下来，比如模型名、样本数、学习率、epoch、显存占用、loss 变化和几条推理对比。

## 下一步可以继续做什么

接下来我们可以继续一起做这几件事里的任意一个：

1. 把你的真实业务数据整理成 `messages` 格式。
2. 给这个项目补一个更正式的 `dataset_builder.py`。
3. 再加一个适合服务器跑的 `run_train.sh`。
4. 按你的显卡规格，反推一套更稳的训练参数。

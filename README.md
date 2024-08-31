
# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

这是一个最简单、最快速的中型 GPT 训练/微调库。它是 [minGPT](https://github.com/karpathy/minGPT) 的重写版本，优先考虑实用性而非教育。仍在积极开发中，但目前 `train.py` 文件可以在 OpenWebText 上复现 GPT-2（124M），在单个 8XA100 40GB 节点上运行约 4 天的训练。代码本身简洁易读：`train.py` 是一个 ~300 行的模板训练循环，而 `model.py` 是一个 ~300 行的 GPT 模型定义，可以选择加载来自 OpenAI 的 GPT-2 权重。就这些。

![repro124m](assets/gpt2_124M_loss.png)

由于代码非常简单，因此很容易根据你的需求进行修改，从头开始训练新模型，或微调预训练的检查点（例如，当前作为起始点的最大模型是 OpenAI 的 GPT-2 1.3B 模型）。


## 安装

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` 用于 huggingface 的 transformers <3（用于加载 GPT-2 检查点）
-  `datasets` 用于 huggingface 的 datasets <3（如果你想下载和预处理 OpenWebText）
-  `tiktoken` 用于 OpenAI 的快速 BPE 代码 <3
-  `wandb` 用于可选的日志记录 <3
-  `tqdm` 用于进度条 <3


## 快速开始

如果你不是深度学习专业人士，只是想感受一下魔力并尝试一下，最快的方法是训练一个字符级的 GPT 来处理莎士比亚的作品。首先，我们将其作为一个单独的（1MB）文件下载，并将其从原始文本转换为一个大型整数流：

```sh
python data/shakespeare_char/prepare.py
```

这会在数据目录中创建一个 `train.bin` 和 `val.bin` 文件。现在是时候训练你的 GPT 了。它的大小非常依赖于你系统的计算资源：

**我有一块 GPU**。太好了，我们可以使用 [config/train_shakespeare_char.py](config/train_shakespeare_char.py) 配置文件中提供的设置快速训练一个小型 GPT：


```sh
python train.py config/train_shakespeare_char.py
```

如果你查看其中的内容，你会看到我们正在训练一个上下文大小最多为 256 个字符、384 个特征通道的 GPT，它是一个 6 层的 Transformer，每层有 6 个头。在一块 A100 GPU 上，这次训练运行大约需要 3 分钟，最佳验证损失为 1.4697。根据配置，模型检查点被写入 `--out_dir` 目录 `out-shakespeare-char`。因此，一旦训练完成，我们可以通过将采样脚本指向这个目录来从最佳模型中进行采样：


```sh
python sample.py --out_dir=out-shakespeare-char
```

这会生成一些样本，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`.在 GPU 上训练 3 分钟后，作为一个字符级模型，效果还不错。通过在这个数据集上微调一个预训练的 GPT-2 模型（请参见稍后的微调部分），很可能会获得更好的结果。

**我只有一台 MacBook**（或其他便宜的电脑）。不用担心，我们仍然可以训练 GPT，但我们需要降低一些要求。我建议使用最新的 PyTorch 每日构建版本（[在这里选择](https://pytorch.org/get-started/locally/) 安装），因为它目前可能会使你的代码更高效。但即使没有，它，简单的训练运行也可以如下所示：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

在这里，由于我们是在 CPU 上运行而不是 GPU，我们必须设置 `--device=cpu` 并且关闭 PyTorch 2.0 编译，使用 `--compile=False`。然后在评估时，我们会得到一个更嘈杂但更快的估计（`--eval_iters=20`，减少自 200），我们的上下文大小仅为 64 个字符而不是 256，批量大小仅为每次迭代 12 个示例，而不是 64。我们还将使用一个更小的 Transformer（4 层、4 个头、128 嵌入大小），并将迭代次数减少到 2000（并相应地通常将学习率衰减到最大迭代次数 `--lr_decay_iters`）。由于我们的网络非常小，我们还会减少正则化（`--dropout=0.0`）。这仍然运行约 3 分钟，但得到的损失仅为 1.88，因此样本质量较差，但仍然很有趣：


```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

生成这样的样本：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

在 CPU 上运行大约 3 分钟的结果还不错，可以给出正确字符风格的一些提示。如果你愿意等待更长时间，可以调整超参数、增加网络规模、上下文长度（`--block_size`）、训练时长等。

最后，在 Apple Silicon Macbook 上使用最新的 PyTorch 版本时，确保添加 `--device=mps`（即 "Metal Performance Shaders" 的缩写）；PyTorch 会使用芯片上的 GPU，这可以 *显著* 加速训练（2-3 倍）并允许你使用更大的网络。更多信息请参见 [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)。


## 复现 GPT-2

更为严肃的深度学习专业人士可能会对复现 GPT-2 的结果更感兴趣。那么我们开始吧——首先我们对数据集进行标记，在这个例子中是 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)，这是 OpenAI （私有） WebText 的开放复现版本：


```sh
python data/openwebtext/prepare.py
```

这将下载并标记 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集。它会创建 `train.bin` 和 `val.bin` 文件，其中包含 GPT2 BPE 标记 ID 的一个序列，存储为原始的 uint16 字节。然后我们就可以开始训练了。要复现 GPT-2（124M），你至少需要一个 8X A100 40GB 的节点，并运行：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

这将在使用 PyTorch 分布式数据并行（DDP）下运行大约 4 天，并将损失降到 ~2.85。现在，一个仅在 OWT 上评估的 GPT-2 模型得到的验证损失约为 3.11，但如果对其进行微调，它将降到 ~2.85 左右（由于明显的领域差距），使得这两种模型大致匹配。

如果你在集群环境中并且拥有多个 GPU 节点，你可以让 GPU 高效运行，例如在 2 个节点上如下所示：


```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

建议对你的互连进行基准测试（例如使用 iperf3）。特别是，如果你没有 Infiniband，则还需要在上述启动命令前添加 `NCCL_IB_DISABLE=1`。你的多节点训练将会正常工作，但很可能会非常缓慢。默认情况下，检查点会定期写入到 `--out_dir`。我们可以通过简单地运行 `python sample.py` 来从模型中进行采样。

最后，要在单个 GPU 上训练，只需运行 `python train.py` 脚本。查看所有参数，该脚本尽力做到非常易读、可修改和透明。根据你的需求，你很可能需要调整其中的一些变量。


## 基线

OpenAI GPT-2 检查点使我们能够为 openwebtext 制定一些基线。我们可以得到如下数字：

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

并观察 train 和 val 上的以下损失：

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

然而，我们必须注意到 GPT-2 是在（闭源，未发布的）WebText 上训练的，而 OpenWebText 只是对这个数据集的最佳开放复现。这意味着存在数据集领域差距。实际上，使用 GPT-2（124M）检查点并直接在 OWT 上微调一段时间后，损失可以降到 ~2.85。这时成为了与复现相关的更合适的基线。

## 微调

微调与训练没有什么不同，我们只需确保从预训练模型初始化，并使用较小的学习率进行训练。有关如何在新文本上微调 GPT 的示例，请进入 `data/shakespeare` 目录并运行 `prepare.py` 以下载小型的 Shakespeare 数据集，并将其渲染为 `train.bin` 和 `val.bin`，使用来自 GPT-2 的 OpenAI BPE 分词器。与 OpenWebText 不同，这将只需几秒钟。微调可以花费很少的时间，例如，在单个 GPU 上只需几分钟。运行一个示例微调如下：


```sh
python train.py config/finetune_shakespeare.py
```

这将加载 `config/finetune_shakespeare.py` 中的配置参数覆盖（不过我并没有调整太多）。基本上，我们从一个 GPT2 检查点初始化，通过 `init_from` 进行训练，训练过程与正常训练类似，只是时间更短且学习率较小。如果你遇到内存不足的问题，可以尝试减小模型大小（可选模型为 `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`）或者可能减少 `block_size`（上下文长度）。最佳检查点（最低验证损失）将位于 `out_dir` 目录中，例如默认情况下是 `out-shakespeare`，根据配置文件进行设置。然后你可以运行 `sample.py --out_dir=out-shakespeare` 中的代码：


```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

哇，GPT，你进入了一个黑暗的地方。我并没有过多调整配置中的超参数，随意尝试吧！

## 采样 / 推理

使用脚本 `sample.py` 可以从 OpenAI 发布的预训练 GPT-2 模型中进行采样，或者从你自己训练的模型中进行采样。例如，以下是从可用的最大 `gpt2-xl` 模型中进行采样的方法：


```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

如果你想从自己训练的模型中进行采样，可以使用 `--out_dir` 来正确指向代码路径。你还可以用文件中的一些文本来提示模型，例如 ```python sample.py --start=FILE:prompt.txt```。

## 效率笔记

对于简单的模型基准测试和性能分析，`bench.py` 可能会很有用。它与 `train.py` 训练循环中的核心部分是相同的，但省略了许多其他复杂性。

请注意，代码默认使用 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。截至编写时（2022年12月29日），这使得 `torch.compile()` 在夜间版本中可用。通过这一行代码的改进是显著的，例如将迭代时间从 ~250ms / iter 降到 135ms / iter。PyTorch 团队做得很棒！

## 待办事项

- 调查并添加 FSDP 替代 DDP
- 在标准评估（如 LAMBADA？HELM？等）上评估零-shot 困惑度
- 调整微调脚本，我认为超参数不太理想
- 在训练过程中安排线性增加批量大小
- 融入其他嵌入（旋转、alibi）
- 将优化器缓冲区与模型参数分开存储在检查点中
- 增加关于网络健康的额外日志记录（如梯度裁剪事件、幅度）
- 对更好的初始化等进行更多调查


## 故障排除

请注意，默认情况下，这个仓库使用 PyTorch 2.0（即 `torch.compile`）。这是一种较新的实验性功能，并且尚未在所有平台上可用（例如 Windows）。如果遇到相关错误消息，可以尝试通过添加 `--compile=False` 标志来禁用这个功能。这会使代码运行速度变慢，但至少能正常运行。

有关此仓库、GPT 和语言建模的一些背景信息，观看我的 [Zero To Hero 系列](https://karpathy.ai/zero-to-hero.html) 可能会有所帮助。特别是，如果你有一定的语言建模背景，[GPT 视频](https://www.youtube.com/watch?v=kCc8FmEb1nY) 可能会很受欢迎。

如果有更多问题或讨论，欢迎访问 Discord 上的 **#nanoGPT** 频道：

[[](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 致谢

所有 nanoGPT 实验均由 [Lambda labs](https://lambdalabs.com) 的 GPU 提供支持，我最喜欢的云 GPU 提供商。感谢 Lambda labs 赞助 nanoGPT！


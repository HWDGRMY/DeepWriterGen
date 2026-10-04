# GitHub - HWDGRMY/DeepWriterGen: 开集在线手写生成，基于预训练书写者识别骨干的整页轨迹扩散模型

## 项目简介

**DeepWriterGen** 是一个面向**开集在线手写生成**的整页轨迹合成系统。它复用 DeepWriterID-v2 的 ConvNeXt V2-Femto 风格编码器，结合 1D U-Net 扩散模型，实现**未见书写者**的整页手写轨迹（.wptt）生成。

本项目解决了在线手写生成领域的核心难题：**如何从少量参考页中提取可迁移的书写风格，并生成结构正确、风格一致的整页文本。** 与闭集识别不同，开集生成要求模型在测试时面对训练中从未见过的书写者，仅凭 4 页参考，就能生成该书写者的整页手写。

> **风格编码器来源**：https://github.com/HWDGRMY/DeepWriterID-v2  
> **数据解析工具**：https://github.com/HWDGRMY/casia-toolkit

---

## 核心特点

- **开集生成**：训练时未见书写者，测试时仅用 4 页参考，即可生成该书写者的整页轨迹。
- **风格-内容解耦**：复用 DeepWriterID-v2 的 512 维风格向量，通过 StyleAdapter 适配生成任务。
- **分层生成**：布局生成器（LSTM）负责字符排列，字符生成器（1D U-Net 扩散）负责单字轨迹。
- **整页时序**：输出完整的 .wptt 时序点序列，含抬笔/落笔标记，可直接用于渲染或进一步分析。
- **零额外标注**：不需要笔画级或部件级标注，仅用 .wptt 原始文件即可训练。

---

## 核心成绩

> **当前状态**：项目框架已搭建完成，训练与评估进行中。

| 指标 | 目标 |
|------|------|
| 生成轨迹 MSE | 待填 |
| DTW（Dynamic Time Warping） | 待填 |
| Content Score（字符正确率） | 待填 |
| Style Score（风格相似度） | 待填 |

**参考基线**：
- **DLG (ICLR 2025)**：分层布局+字形生成，在 CASIA-OLHWDB 上验证了开集生成可行性。
- **DNA (WACV 2026)**：双分支网络，在未见书写者和未见字符任务上达到 SOTA。
- **SDT (CVPR 2023)**：风格解耦 Transformer，Style Score 达 94.5%。

---

## 📊 数据集说明

本项目使用 **CASIA-OLHWDB 2.0-2.2 在线手写文本数据集**，与 DeepWriterID-v2 完全对齐。

对比维度 | 说明 |
|---|---|
| **作者数量** | **1019 位** 书写者 |
| **每位作者页数** | **5 页**（4 页训练参考 + 1 页测试目标） |
| **文本内容** | **内容各不相同**（不同新闻、古诗模板） |
| **训练样本量** | 约 **120 万** 个动态增强伪字符 |

**数据划分**：
- **训练集**：700 位书写者（每位 5 页）
- **验证集**：150 位书写者（每位 5 页）
- **测试集**：169 位书写者（每位 5 页）

> **注**：训练/验证/测试的书写者完全不重叠，符合开集生成设定。

---

## 📁 项目目录结构

```
DeepWriterGen/
├── configs/                            # 参数配置
│   ├── convnext.yaml                   # 风格编码器配置（复用 v2）
│   └── diffusion.yaml                  # 扩散生成器超参数
├── data/
│   ├── raw/                            # 原始 .wptt 轨迹
│   ├── features/
│   │   ├── metadata.csv                # 数据划分（复用 v2）
│   │   ├── char_map.json               # 字符到 ID 映射
│   │   └── style_vectors.pt            # 预计算的风格向量
│   └── page_cache/                     # 整页伪字符/布局缓存
├── src/
│   ├── data/
│   │   ├── loader.py                   # 解析 .wptt（含结构信息）
│   │   ├── page_dataset.py             # 整页轨迹数据集
│   │   └── layout_utils.py             # 轨迹重采样、bbox 计算
│   ├── models/
│   │   ├── backbone.py                 # ConvNeXt V2-Femto（复用 v2）
│   │   ├── style_adapter.py            # 风格向量适配 MLP
│   │   ├── layout_generator.py         # LSTM 布局生成器
│   │   ├── diffusion_unet.py           # 1D U-Net 去噪器
│   │   └── diffusion_sampler.py        # DDIM 采样
│   ├── preprocessing/
│   │   ├── corner.py                   # 拐点检测（复用 v2）
│   │   └── segmentation.py             # 伪字符切分（复用 v2）
│   ├── features/
│   │   └── path_signature.py           # 路径签名（复用 v2）
│   ├── augmentation/
│   │   └── drop_segment.py             # DropSegment（复用 v2）
│   ├── training/
│   │   └── trainer_gen.py              # 生成模型训练引擎
│   └── evaluation/
│       └── evaluator_gen.py            # DTW/Content/Style 评估
├── scripts/
│   ├── preprocess.py                   # 数据预处理（复用 v2）
│   ├── preprocess_gen.py               # 生成字符映射
│   ├── precompute_style.py             # 预计算风格向量
│   ├── train_gen.py                    # 生成模型训练
│   └── evaluate_gen.py                 # 生成模型评估
├── outputs/
│   ├── logs/                           # 训练日志 CSV
│   └── checkpoints/                    # 模型权重 .pth
├── pretrained/                         # 预训练权重
├── environment.yml                     # Conda 环境配置
├── setup.py                            # 包安装配置
├── README.md                           # 项目说明
└── LICENSE                             # MIT 许可证
```

---

## 🚀 快速开始

### 1. 环境配置

```bash
conda create -n deepwritergen python=3.9 -y
conda activate deepwritergen

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install timm signatory numpy pandas opencv-python tqdm pyyaml
pip install signatory --no-build-isolation  # 若编译失败
```

### 2. 数据准备

从 CASIA 官网下载 OLHWDB 2.0-2.2：

```bash
python scripts/preprocess.py
```

生成 `data/features/metadata.csv`，按书写者划分训练/测试集。

### 3. 训练风格编码器（可选）

若已有 DeepWriterID-v2 的 `convnext_best.pth`，可跳过此步；否则先训练：

```bash
python scripts/train.py --config configs/convnext.yaml
```

### 4. 预计算风格向量

```bash
python scripts/precompute_style.py
```

生成 `data/features/style_vectors.pt`。

### 5. 构建字符映射

```bash
python scripts/preprocess_gen.py
```

生成 `data/features/char_map.json`。

### 6. 训练生成模型

```bash
python scripts/train_gen.py
```

**特性**：
- 冻结风格编码器，只训生成器
- Warmup + 余弦退火
- 早停（验证集连续 20 轮未提升）
- 最佳模型保存

### 7. 评估

```bash
python scripts/evaluate_gen.py
```

计算 DTW、Content Score、Style Score。

---

## 🧠 方法学说明

### 1. 开集设定

训练时用 **700 位书写者**，验证用 **150 位**，测试用 **169 位**。三者完全不重叠。

测试时，每位未见书写者提供 **4 页参考**，模型生成第 **5 页** 的整页轨迹。

### 2. 分层生成架构

```
输入：文本内容 C + 风格参考 X_ref（4 页 .wptt）
  ↓
Layer 1: Layout Generator（LSTM）
  - 输入：字符序列 + 风格参考的 bbox 前缀
  - 输出：每个字符的 bounding box
  ↓
Layer 2: Character Generator（1D U-Net 扩散）
  - 条件：风格向量 z_s + 字符内容 c_i + bbox
  - 输出：该字符的轨迹点序列
  ↓
拼接：按布局顺序拼接所有字符轨迹，输出整页 .wptt
```

### 3. 风格向量提取

复用 DeepWriterID-v2 的 `ConvNeXtBackbone.forward(x, return_features=True)`：

- 输入：伪字符特征图（位图 + 路径签名，64 通道）
- 输出：512 维风格向量
- 对每位书写者的所有页、所有伪字符取平均，得到该书写者的风格向量

### 4. 过拟合判断

**当前状态**：训练与评估进行中，具体过拟合判断待训练完成后补充。

**预期策略**：
- DropPath 0.1 + Dropout 0.3/0.4
- DropSegment 数据增强
- 早停（验证集连续 20 轮未提升）
- 最佳模型保存

---

## 📉 数据集划分与验证集说明

### 1. 划分策略

本项目采用 **按书写者划分**：
- 训练集：700 位书写者
- 验证集：150 位书写者
- 测试集：169 位书写者

每位书写者保留全部 5 页，训练/验证时用 4 页参考、1 页目标。

### 2. 与 DeepWriterID-v2 的区别

DeepWriterID-v2 未划分独立验证集，每轮在测试集上评估并保存最佳模型，存在选择偏差。

本项目**引入了独立验证集**，用于早停、选模型、调超参，测试集只在最后用一次，从而获得更接近无偏的泛化估计。

---

## 📦 模型文件说明

模型权重不包含在 GitHub 仓库中，原因：

1. **算力成本**：训练消耗大量云端算力
2. **模型资产保护**：风格编码器基于 DeepWriterID-v2，生成器为独立训练成果

### 📎 如何获取

- **GitHub Issues**：新建 Issue 说明用途
- **邮件**：`zhouhao_oss@163.com`，附身份、用途及具体场景

> **注意**：模型文件仅限申请用途使用，请勿二次分发。

---

## 📜 第三方代码与引用

### DeepWriterID-v2（风格编码器）

- **仓库**：https://github.com/HWDGRMY/DeepWriterID-v2
- **许可证**：MIT

### casia-toolkit（数据解析）

- **仓库**：https://github.com/HWDGRMY/casia-toolkit
- **许可证**：MIT

### timm（PyTorch Image Models）

- **仓库**：https://github.com/huggingface/pytorch-image-models
- **许可证**：Apache License 2.0

### ConvNeXt V2

- **论文**：Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023
- **仓库**：https://github.com/facebookresearch/ConvNeXt-V2
- **许可证**：MIT（代码）

### DLG (ICLR 2025)

- **论文**：Ren et al., *Decoupling Layout from Glyph in Online Chinese Handwriting Generation*, ICLR 2025
- **参考**：分层布局+字形生成架构

### DNA (WACV 2026)

- **论文**：Huang et al., *DNA: Dual-branch Network with Adaptation for Open-Set Online Handwriting Generation*, WACV 2026
- **参考**：双分支风格-内容解耦

### SDT (CVPR 2023)

- **论文**：Dai et al., *Disentangling Writer and Character Styles for Handwriting Generation*, CVPR 2023
- **参考**：WriterNCE 对比学习

---

## 🙏 致谢

- 数据集：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。
- 风格编码器：DeepWriterID-v2 (ConvNeXt V2-Femto)。
- 数据解析：casia-toolkit。
- 生成架构参考：DLG (ICLR 2025)、DNA (WACV 2026)、SDT (CVPR 2023)。

---

## 💬 反馈与建议

如果你在使用本项目的过程中遇到任何问题，或者有更好的改进思路（比如更高效的风格解耦、更优的采样策略等），非常欢迎你在 GitHub 上提交 **Issue** 或直接发起 **Pull Request**。

**其他联系方式**：也可以通过作者邮箱 `zhouhao_oss@163.com` 与我沟通。

我可能不会及时回复每一条消息，但所有有价值的建议都会认真考虑，并用于后续的迭代和优化。如果你在跑这个项目时卡在了某个环节，也欢迎在 Issues 里提问。

---

## 📄 许可证

本项目采用 MIT 许可证。
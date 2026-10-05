# DeepWriterGen

## 项目简介

**DeepWriterGen** 是一个面向 **开集在线手写生成** 的整页轨迹合成框架。它复用 DeepWriterID-v2 的 ConvNeXt V2-Femto 作为风格编码器，结合 1D U-Net 扩散模型，实现未见书写者的整页手写轨迹（.wptt）生成。

本项目的核心思路：
- **行级生成**：从整页 .wptt 中拆出每行，以行为单位训练和生成，避免字符级切分难题。
- **字宽归一化**：以平均字宽为归一化单位，使模型学到与物理尺寸无关的轨迹形状。
- **用户可控分行**：不训练布局模型，推理时用户指定分行（或按字数自动分行）。
- **开集设定**：训练、验证、测试使用完全不同的书写者。

> **风格编码器来源**：https://github.com/HWDGRMY/DeepWriterID-v2  
> **数据解析工具**：https://github.com/HWDGRMY/casia-toolkit

---

## 核心特点

- **开集生成**：训练时未见书写者，测试时仅用其风格向量即可生成该书写者的轨迹。
- **行级扩散模型**：1D U-Net 条件扩散，参数约 23M，训练稳定。
- **字宽归一化**：坐标除以平均字宽，一个字跨度 ≈ 1.0，模型学形状而非像素。
- **灵活的推理接口**：支持手动指定分行，或按每行字数自动分行。
- **兼容 .wptt**：生成的归一化轨迹可通过后处理转为 .wptt 格式。

---

## 当前状态

- ✅ 数据预处理（`preprocess_gen.py`）：生成 `metadata_gen.csv` 和 `char_map.json`
- ✅ 风格向量提取（`precompute_style.py`）：多进程提取 1019 位书写者
- ✅ Line Diffusion 训练（`train_gen.py`）：训练框架完成，正在训练中
- ⏳ 定量评估（DTW / Content Score / Style Score）：待训练完成后补充
- ⏳ 生成可视化（`generate_page.py`）：已提供生成和可视化脚本

---

## 数据集说明

使用 **CASIA-OLHWDB 2.0-2.2** 在线手写文本数据集。

| 项 | 值 |
|---|---|
| 书写者总数 | 1019 位 |
| 每位书写者页数 | 约 5 页 |
| 划分方式 | 按书写者不重叠划分 |
| 训练集 | 700 位（3497 页） |
| 验证集 | 150 位（750 页） |
| 测试集 | 169 位（845 页） |
| 预处理后总行数 | 约 36000 行（训练）|

## 项目目录结构

```
DeepWriterGen/
├── configs/
│   └── diffusion.yaml              # 扩散模型超参数
├── data/
│   ├── raw/                        # 原始 .wptt（需自行下载）
│   ├── features/
│   │   ├── metadata_gen.csv        # 数据划分（已上传）
│   │   ├── char_map.json           # 字符表（不上传）
│   │   └── style_vectors.pt        # 风格向量（不上传）
│   └── page_cache/                 # 预处理缓存（可选）
├── src/
│   ├── data/
│   │   ├── loader.py               # .wptt 解析（行结构）
│   │   └── line_dataset.py         # 行级数据集
│   ├── models/
│   │   ├── backbone.py             # ConvNeXt V2 风格编码器（复用 v2）
│   │   ├── diffusion_unet.py       # 1D U-Net 去噪器
│   │   ├── diffusion_sampler.py    # DDIM 采样
│   │   └── style_adapter.py        # 风格向量适配层
│   ├── preprocessing/
│   │   ├── corner.py               # 拐点检测
│   │   └── segmentation.py         # 伪字符切分
│   ├── features/
│   │   └── path_signature.py       # 路径签名特征
│   └── training/
│       └── trainer_gen.py          # 生成模型训练引擎
├── scripts/
│   ├── preprocess_gen.py           # 数据预处理
│   ├── precompute_style.py         # 提取风格向量
│   ├── train_gen.py                # 训练入口
│   ├── generate_page.py            # 整页生成
│   └── test_layout.py              # (已弃用) Layout 测试
├── outputs/
│   ├── checkpoints/                # 模型权重
│   └── generated/                  # 生成结果
├── environment.yml
├── setup.py
├── README.md
└── .gitignore
```

---

## 快速开始

### 1. 环境配置

```bash
conda create -n deepwritergen python=3.10 -y
conda activate deepwritergen

# 安装 PyTorch（根据 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install timm signatory numpy pandas opencv-python tqdm pyyaml
pip install signatory --no-build-isolation  # 若编译失败
```

### 2. 数据准备

从 CASIA 官网下载 OLHWDB 2.0-2.2，解压到 `data/raw/`。目录结构如下：

```
data/raw/
├── WPTT2.0-Train/
├── WPTT2.0-Test/
├── WPTT2.1-Train/
├── WPTT2.1-Test/
├── WPTT2.2-Train/
└── WPTT2.2-Test/
```

运行预处理脚本：

```bash
python scripts/preprocess_gen.py
```

生成 `data/features/metadata_gen.csv`（已上传）和 `char_map.json`（需自行生成）。

### 3. 提取风格向量

需要先准备 DeepWriterID-v2 训练好的风格编码器 `convnext_best.pth`，放在 `outputs/checkpoints/` 下。

```bash
python scripts/precompute_style.py
```

生成 `data/features/style_vectors.pt`，包含 1019 位书写者的 512 维风格向量。

### 4. 训练 Line Diffusion

```bash
python scripts/train_gen.py
```

训练参数在 `configs/diffusion.yaml` 中配置。默认：
- 100 epochs
- batch size 32
- 学习率 1e-4
- 早停 patience 15

训练日志输出到终端，最佳模型保存至 `outputs/checkpoints/gen_best.pth`。

### 5. 生成整页

```bash
python scripts/generate_page.py
```

默认示例：使用第一位书写者的风格生成两行七言诗。可在脚本中修改：

```python
# 方式 1：直接指定分行
lines = [
    '落霞与孤鹜齐飞',
    '秋水共长天一色',
]

# 方式 2：给定文本 + 每行字数
text = '落霞与孤鹜齐飞秋水共长天一色'
lines = auto_wrap(text, chars_per_line=7)
```

生成结果保存为 `outputs/generated/page.npy` 和 `page.png`。

---

## 模型架构

### 风格编码器（冻结）

- 复用 DeepWriterID-v2 的 ConvNeXt V2-Femto 骨干
- 输出 512 维风格向量
- 输入：伪字符的位图 + 路径签名特征图（64 通道）

### Line Diffusion（训练）

- 1D U-Net 去噪器，参数约 23M
- 条件：风格向量（StyleAdapter 适配）+ 行文本嵌入（双向 LSTM）
- 输入：噪声轨迹 (B, 2, L)，L=256
- 输出：预测噪声
- 扩散步数：1000，DDIM 采样步数：50

---

## 引用

如果本工作对你有帮助，请引用相关论文与仓库：

```bibtex
@misc{deepwriterid_v2,
  author = {HWDGRMY},
  title = {DeepWriterID v2.0: ConvNeXt-based writer identification},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/HWDGRMY/DeepWriterID-v2}}
}
```

相关论文：
- DLG: Ren et al., *Decoupling Layout from Glyph in Online Chinese Handwriting Generation*, ICLR 2025.
- Diff-Font: He et al., *Diff-Font: Diffusion Model for Robust One-Shot Font Generation*, IJCV 2024.
- SDT: Dai et al., *Disentangling Writer and Character Styles for Handwriting Generation*, CVPR 2023.

---

## 致谢

- 原始论文：Weixin Yang, Lianwen Jin, et al. *DeepWriterID: An End-to-end Online Text-independent Writer Identification System*.
- 数据集：中国科学院自动化研究所 CASIA-OLHWDB 手写数据库。
- 风格编码器：DeepWriterID-v2 (ConvNeXt V2-Femto)。
- 数据解析：casia-toolkit。
- 扩散模型参考：DLG (ICLR 2025)、Diff-Font (IJCV 2024)。

---

## 反馈与建议

如果你在使用本项目的过程中遇到任何问题，或者有更好的改进思路，欢迎在 GitHub 上提交 **Issue** 或 **Pull Request**。

**其他联系方式**：`zhouhao_oss@163.com`

---

## 许可证

本项目采用 MIT 许可证。
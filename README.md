# StyleSlip

StyleSlip 是从 TextWave 中独立拆出的 **StyleDistance leave-one-out（Style-LOO）**
特征提取器。它直接读取原始 JSONL/CSV 文本，不再要求先生成 WordNet 基础归档，也不依赖
TextWave 的 Python 包。

每个含字母的 token 会从所在句子（或长句分块）中删除并重新编码，生成三条固定长度通道：

1. `loo_cosine_distance_half`：原上下文与删除后上下文的半余弦距离；
2. `loo_direction_alignment`：删除方向与文档平均删除方向的对齐度；
3. `loo_direction_drift_half`：相邻删除方向之间的半余弦距离。

默认处理完整文档。`--context-tokens` 只限制 StyleDistance 的局部上下文长度；只有显式设置
`--max-tokens` 才会截断文档。

## 复用 TextWave 环境

无需新建虚拟环境，也无需修改现有环境。本机已有解释器可直接运行零安装入口：

```powershell
& C:\Users\three\.conda\envs\TextWave\python.exe .\run.py --help
```

该环境已经包含 `sentence-transformers`。如果希望注册 `styleslip` 命令，可选择进行 editable
安装；这不会复制或重建环境：

```powershell
& C:\Users\three\.conda\envs\TextWave\python.exe -m pip install -e . --no-build-isolation
```

若在别的机器安装完整模型依赖，可使用：

```powershell
python -m pip install -e ".[model]"
```

StyleDistance 模型和 NLTK 分句资源仍需已缓存。两个项目保持当前同级布局时会自动发现
`..\Textwave\.nltk_data`；也可显式指定该目录：

```powershell
$python = "C:\Users\three\.conda\envs\TextWave\python.exe"
& $python .\run.py `
  --input examples\sample.jsonl `
  --output outputs\sample_style_loo.npz `
  --nltk-data ..\Textwave\.nltk_data
```

默认 `--local-files-only`，防止运行时意外联网。首次确实需要下载模型时显式传入
`--no-local-files-only`。

## 输入格式

JSONL 和 CSV 默认读取 `text` 与可选的 `label` 字段；标签可以是 `0/1` 或常见的
`human/ai` 字符串。也会保留可选的 `id`、`model`、`source/domain`：

```json
{"id":"doc-1","text":"Example document text.","label":0,"model":"human","source":"news"}
```

字段名不同时可用 `--text-column` 和 `--label-column` 指定。标签必须全部存在或全部缺省。

输出 NPZ 的核心数组为：

- `waves`: `[documents, 3, wave_length]` 的 `float32` 数组；
- `channel_names`: 三个通道的固定名称；
- `labels`: 输入有标签时写入；
- `ids/models/sources/fingerprints`: 样本元数据；
- `style_*`: token 数、上下文数、有效删除数与方向集中度；
- `config_json`: 完整提取配置。

该布局与 TextWave 的训练数据约定兼容。

## 验证

```powershell
& C:\Users\three\.conda\envs\TextWave\python.exe -m pytest
```

测试使用确定性的假编码器，不下载模型，也不占用 GPU。

## 数据集实验入口

统一实验入口是 `experiment.py`。它只启用本项目需要的三类协议：

| `--dataset` | `--scenario` | 数据与划分协议 |
| --- | --- | --- |
| `m4` | `monolingual` | Subtask A 单语言；官方 train/dev/test |
| `m4` | `multilingual` | Subtask A 多语言；官方 train/dev/test |
| `m4` | `both` | 顺序运行上述两个实验 |
| `deepfake` | `cross_domains_cross_models` | 官方 train/valid/test，并评估 `test_ood*.csv` |
| `deepfake` | `unseen_models` | 每个 `unseen_model_*` 子目录作为独立 case |
| `deepfake` | `unseen_domains` | 每个 `unseen_domain_*` 子目录作为独立 case |
| `deepfake` | `all` | 只运行上述三类 Deepfake 场景 |
| `raid` | `clean` | `train_none.csv` 内部分组切分，`extra_none.csv` 做 OOD |
| `raid` | `attacked` | `train.csv` 内部分组切分，`extra.csv` 做 OOD |
| `raid` | `both` | 顺序运行 clean 与 attacked |

兼容别名包括 `bilingual → multilingual`、`no_attack → clean`、`attack → attacked`；
`cross_domains_corss_models` 这一常见拼写也会规范到实际目录名
`cross_domains_cross_models`。

先检查程序发现的文件，不读取正文或加载模型：

```powershell
$python = "C:\Users\three\.conda\envs\TextWave\python.exe"
& $python .\experiment.py --dataset m4 --list
& $python .\experiment.py --dataset deepfake --scenario unseen_models --list
& $python .\experiment.py --dataset raid --list
```

常用运行示例：

```powershell
# M4 Subtask A 单语言
& $python .\experiment.py --dataset m4 --scenario monolingual

# M4 Subtask A 单语言和多语言
& $python .\experiment.py --dataset m4 --scenario both

# Deepfake 跨领域、跨模型
& $python .\experiment.py --dataset deepfake --scenario cross_domains_cross_models

# Deepfake unseen model；只跑目录名包含 bloom 的 case
& $python .\experiment.py --dataset deepfake --scenario unseen_models --case bloom

# Deepfake 所有 unseen domain case
& $python .\experiment.py --dataset deepfake --scenario unseen_domains

# RAID 无攻击和有攻击实验
& $python .\experiment.py --dataset raid --scenario both
```

默认每个 split、每个类别确定性抽样 500 条。Style-LOO 需要为每个 token 重复编码，正式规模应按
显存和时间逐步增加，例如：

```powershell
& $python .\experiment.py --dataset raid --scenario clean `
  --samples-per-class 5000 --ood-samples-per-class 5000
```

RAID 的 `test.csv/test_none.csv` 只有 `id,generation`，属于盲测数据。程序只在 manifest 中记录
其路径并设置 `blind_test_evaluated=false`，不会推断标签或计算指标。validation 和内部 test
均从对应 train 文件按 `source_id` 分组切分；同一原文及其攻击版本不会跨 split。划分索引保存为
`split_indices.npz`。`extra*.csv` 只用于训练完成后的 OOD 测试。

Deepfake 源文件使用 `label=1` 表示 human、`label=0` 表示 AI；专用加载器会校验
`is_human` 后统一转换为本项目的 `human=0/AI=1`。M4 保持原始 Subtask A 标签。

### 特征缓存与实验输出

默认目录完全分离：

```text
artifacts/features/                 # 可跨实验复用的 Style-LOO NPZ
  <dataset>/<hash-prefix>/<hash>.npz
  <dataset>/<hash-prefix>/<hash>.json

outputs/experiments/                # 与训练参数相关的模型和指标
  <dataset>/<scenario>/<case>/<run-hash>/
    experiment_manifest.json
    best_model.pt
    metrics.json
    validation_predictions.csv
    test_predictions.csv
    ood_predictions.csv             # 存在 OOD split 时
    split_indices.npz               # RAID 内部划分时
  results_<dataset>.csv
  results_<dataset>.json
```

特征缓存键包含源文件路径、大小、修改时间、采样数、随机种子、去重策略、StyleDistance 模型身份和
全部 Style-LOO 参数。改变 TCN 深度、epoch、batch size 或输出目录不会重新提取特征；输入文件或
Style-LOO 参数变化则生成新的缓存条目，不覆盖旧结果。

```powershell
# 只构建/复用特征，不训练
& $python .\experiment.py --dataset m4 --scenario both --prepare-only

# 显示将使用的输入和缓存路径
& $python .\experiment.py --dataset raid --scenario attacked --dry-run

# 强制重建特征，或只重跑训练
& $python .\experiment.py --dataset m4 --scenario monolingual --refresh-feature-cache
& $python .\experiment.py --dataset m4 --scenario monolingual --refresh-results
```

官方预定义 split 默认使用 `--overlap-policy error` 检查归一化文本泄漏。确认要保留时使用
`allow`，希望按 train → validation → test 的优先级删除后续重复文本时使用 `drop`。

### 训练进度

训练默认显示 case 和 epoch 进度。epoch 进度条实时包含：

- 当前/总 epoch；
- `train` loss 和 `val` loss；
- 最佳 validation loss 及其 epoch；
- 当前早停计数 `patience`。

需要观察每个 batch 时增加 `--batch-progress`：

```powershell
& $python .\experiment.py --dataset m4 --scenario monolingual --batch-progress
```

batch 进度条会显示即时 loss 和已处理样本数。非交互日志环境可以使用 `--no-progress` 关闭动态
进度条；训练结果和逐 epoch 记录仍会写入：

```text
training_history.csv    # 每个 epoch 的损失、最佳轮次、早停计数和耗时
training_status.json    # 训练期间实时更新；结束后包含 completed/test/OOD 摘要
metrics.json            # 最终完整指标与训练历史
```

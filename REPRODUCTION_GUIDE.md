# GDP-Zero 论文复现操作指南

本文档提供完整的论文复现步骤。

## 一、环境准备

### 1.1 安装 Python 依赖

根据代码分析，需要安装以下依赖包：

```bash
pip install numpy
pip install tqdm
pip install openai
pip install transformers
pip install torch
pip install nltk
pip install tenacity
```

或者创建 `requirements.txt` 后统一安装：

```bash
pip install -r requirements.txt
```

### 1.2 设置环境变量

**必须设置 OpenAI API Key**（二选一）：

**选项A：使用 OpenAI 官方 API**
```bash
export OPENAI_API_KEY="sk-ejyFNvi45V3BzCgKOEHhN8mc0Qt8lLS9xpZPSLZVeCgCADKl"  # 替换为你的实际 API Key
```

**选项B：使用 Azure OpenAI**
```bash
export MS_OPENAI_API_KEY="xxxx"
export MS_OPENAI_API_BASE="https://xxx.com"
export MS_OPENAI_API_VERSION="xxx"
export MS_OPENAI_API_CHAT_VERSION="xxx"
```

### 1.3 设置 PYTHONPATH

**非常重要**：必须将项目根目录添加到 PYTHONPATH

```bash
cd GDPZero
export PYTHONPATH=$(pwd)
```

或者添加到 `~/.bashrc` 使其永久生效：
```bash
echo 'export PYTHONPATH=/root/autodl-tmp/MCT/GDPZero:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### 1.4 下载 NLTK 数据（如果需要）

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 二、数据准备

### 2.1 检查数据文件

确认数据文件存在：
```bash
ls -lh data/p4g/300_dialog_turn_based.pkl
```

如果文件不存在，需要从论文作者提供的链接下载 P4G 数据集。

## 三、运行实验

### 3.1 GDP-Zero 主实验

根据 README，论文中的主要实验配置如下：

**使用 gpt-3.5-turbo，n=10 次模拟，k=3 个实现，Q_0=0.25：**
```bash
python runners/gdpzero.py \
    --output outputs/gdpzero_10sims_3rlz_0.25Q0_20dialogs.pkl \
    --llm gpt-3.5-turbo \
    --num_mcts_sims 10 \
    --max_realizations 3 \
    --Q_0 0.25 \
    --num_dialogs 20
```

**使用 gpt-3.5-turbo，n=20 次模拟，k=3 个实现，Q_0=0.0：**
```bash
python runners/gdpzero.py \
    --output outputs/gdpzero_20sims_3rlz_0.0Q0_20dialogs.pkl \
    --llm gpt-3.5-turbo \
    --num_mcts_sims 20 \
    --max_realizations 3 \
    --Q_0 0.0 \
    --num_dialogs 20
```

**使用 gpt-3.5-turbo，n=50 次模拟，k=3 个实现，Q_0=0.0：**
```bash
python runners/gdpzero.py \
    --output outputs/gdpzero_50sims_3rlz_0.0Q0_20dialogs.pkl \
    --llm gpt-3.5-turbo \
    --num_mcts_sims 50 \
    --max_realizations 3 \
    --Q_0 0.0 \
    --num_dialogs 20
```

### 3.2 Baseline 实验（Raw Prompting）

```bash
python runners/raw_prompting.py \
    --output outputs/chatgpt_raw_prompt.pkl \
    --llm gpt-3.5-turbo
```

### 3.3 Ablation 实验

**无 OpenLoop 版本：**
```bash
python runners/gdpzero_noopenloop.py \
    --output outputs/gdpzero_noopenloop.pkl \
    --llm gpt-3.5-turbo \
    --num_mcts_sims 10
```

**无 Response Selection 版本：**
```bash
python runners/gdpzero_noRS.py \
    --output outputs/gdpzero_noRS.pkl \
    --llm gpt-3.5-turbo \
    --num_mcts_sims 10 \
    --max_realizations 3 \
    --Q_0 0.25
```

## 四、评估结果

### 4.1 与人类回复对比

使用 GPT-3.5-turbo 作为评判者，比较模型生成回复与人类回复：

```bash
python test.py \
    -f outputs/gdpzero_10sims_3rlz_0.25Q0_20dialogs.pkl \
    --output outputs/eval_vs_human.pkl \
    --judge gpt-3.5-turbo
```

预期输出示例：
```
evaluating: 100%|███████████████| 154/154 [03:49<00:00,  1.49s/it]
win rate: 93.51%
stats:  {'win': 144, 'draw': 0, 'lose': 10}
```

### 4.2 Head-to-Head 对比

比较 GDP-Zero 与 Baseline（Raw Prompting）：

```bash
python test.py \
    -f outputs/gdpzero_10sims_3rlz_0.25Q0_20dialogs.pkl \
    --h2h outputs/chatgpt_raw_prompt.pkl \
    --output outputs/eval_h2h.pkl \
    --judge gpt-3.5-turbo
```

预期输出示例：
```
evaluating: 100%|███████████████| 154/154 [03:29<00:00,  1.36s/it]
win rate: 59.09%
stats:  {'win': 91, 'draw': 2, 'lose': 61}
```

## 五、完整复现流程（推荐顺序）

### 步骤 1：环境检查
```bash
# 检查 Python 版本（建议 3.7+）
python --version

# 检查 API Key 是否设置
echo $OPENAI_API_KEY  # 或 echo $MS_OPENAI_API_KEY

# 检查 PYTHONPATH
echo $PYTHONPATH
```

### 步骤 2：快速测试（小规模）
```bash
# 先用小参数测试是否能正常运行
python runners/gdpzero.py \
    --output outputs/test.pkl \
    --llm gpt-3.5-turbo \
    --num_mcts_sims 5 \
    --max_realizations 2 \
    --Q_0 0.25 \
    --num_dialogs 2 \
    --debug
```

### 步骤 3：运行完整实验
按照第三部分的命令依次运行所有实验配置。

### 步骤 4：评估所有结果
按照第四部分对每个实验结果进行评估。

## 六、常见问题排查

### 问题 1：ModuleNotFoundError
**原因**：PYTHONPATH 未设置
**解决**：
```bash
export PYTHONPATH=$(pwd)
```

### 问题 2：API Key 错误
**原因**：环境变量未正确设置
**解决**：
```bash
# 检查环境变量
env | grep OPENAI
# 重新设置
export OPENAI_API_KEY="sk-xxxx"
```

### 问题 3：数据文件不存在
**原因**：P4G 数据集未下载
**解决**：从论文作者提供的链接下载 `300_dialog_turn_based.pkl` 并放到 `data/p4g/` 目录

### 问题 4：API 速率限制
**原因**：OpenAI API 有速率限制
**解决**：
- 代码中已使用 `tenacity` 进行重试和指数退避
- 可以降低 `--num_mcts_sims` 参数减少 API 调用
- 考虑使用 Azure OpenAI 可能有更高的速率限制

### 问题 5：内存不足
**原因**：MCTS 搜索树占用内存
**解决**：
- 减少 `--num_mcts_sims` 参数
- 减少 `--max_realizations` 参数
- 减少 `--num_dialogs` 参数

## 七、预期结果

根据论文和 README，预期结果应该接近：

- **GDP-Zero vs Human**: Win rate ~93-95%
- **GDP-Zero vs Raw Prompting**: Win rate ~55-60%

## 八、时间估算

- **单次实验（20个对话，10次模拟）**：约 1-2 小时（取决于 API 响应速度）
- **完整复现（所有配置）**：约 5-10 小时
- **评估阶段**：约 1-2 小时

## 九、输出文件说明

所有输出文件为 pickle 格式，包含：
- `did`: 对话 ID
- `context`: 对话上下文
- `ori_resp`: 原始回复（人类或 baseline）
- `ori_da`: 原始对话行为
- `new_resp`: 模型生成回复
- `new_da`: 模型选择的对话行为
- `debug`: MCTS 搜索树统计信息

## 十、注意事项

1. **API 费用**：大量实验会产生较高的 API 调用费用，请提前估算
2. **断点续跑**：代码支持断点续跑，每完成一个对话就会保存结果
3. **随机性**：由于使用采样，每次运行结果可能略有不同
4. **版本兼容性**：确保 OpenAI API 版本兼容（代码使用较旧的 API，可能需要调整）


#!/bin/bash
# GDP-Zero 快速启动脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "GDP-Zero 论文复现快速启动脚本"
echo "=========================================="

# 1. 检查环境变量
echo "[1/5] 检查环境变量..."
if [ -z "$OPENAI_API_KEY" ] && [ -z "$MS_OPENAI_API_KEY" ]; then
    echo "错误: 未设置 OPENAI_API_KEY 或 MS_OPENAI_API_KEY"
    echo "请运行: export OPENAI_API_KEY='sk-xxxx'"
    exit 1
fi
echo "✓ API Key 已设置"

# 2. 设置 PYTHONPATH
echo "[2/5] 设置 PYTHONPATH..."
export PYTHONPATH=$(pwd)
echo "✓ PYTHONPATH=$PYTHONPATH"

# 3. 检查依赖
echo "[3/5] 检查 Python 依赖..."
python -c "import numpy, tqdm, openai, transformers, torch, nltk, tenacity" 2>/dev/null || {
    echo "警告: 部分依赖缺失，请运行: pip install -r requirements.txt"
    read -p "是否现在安装依赖? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    else
        echo "跳过依赖安装，继续..."
    fi
}

# 4. 检查数据文件
echo "[4/5] 检查数据文件..."
if [ ! -f "data/p4g/300_dialog_turn_based.pkl" ]; then
    echo "警告: 数据文件不存在: data/p4g/300_dialog_turn_based.pkl"
    echo "请确保已下载 P4G 数据集"
    exit 1
fi
echo "✓ 数据文件存在"

# 5. 选择运行模式
echo "[5/5] 选择运行模式..."
echo ""
echo "请选择要运行的实验:"
echo "1) GDP-Zero (n=10, k=3, Q_0=0.25) - 推荐用于快速测试"
echo "2) GDP-Zero (n=20, k=3, Q_0=0.0)"
echo "3) GDP-Zero (n=50, k=3, Q_0=0.0) - 完整实验"
echo "4) Baseline (Raw Prompting)"
echo "5) 评估已有结果 (vs Human)"
echo "6) 评估已有结果 (Head-to-Head)"
echo "7) 交互式演示"
echo ""
read -p "请输入选项 (1-7): " choice

case $choice in
    1)
        echo "运行 GDP-Zero (n=10, k=3, Q_0=0.25)..."
        python runners/gdpzero.py \
            --output outputs/gdpzero_10sims_3rlz_0.25Q0_20dialogs.pkl \
            --llm gpt-3.5-turbo \
            --num_mcts_sims 10 \
            --max_realizations 3 \
            --Q_0 0.25 \
            --num_dialogs 20
        ;;
    2)
        echo "运行 GDP-Zero (n=20, k=3, Q_0=0.0)..."
        python runners/gdpzero.py \
            --output outputs/gdpzero_20sims_3rlz_0.0Q0_20dialogs.pkl \
            --llm gpt-3.5-turbo \
            --num_mcts_sims 20 \
            --max_realizations 3 \
            --Q_0 0.0 \
            --num_dialogs 20
        ;;
    3)
        echo "运行 GDP-Zero (n=50, k=3, Q_0=0.0)..."
        python runners/gdpzero.py \
            --output outputs/gdpzero_50sims_3rlz_0.0Q0_20dialogs.pkl \
            --llm gpt-3.5-turbo \
            --num_mcts_sims 50 \
            --max_realizations 3 \
            --Q_0 0.0 \
            --num_dialogs 20
        ;;
    4)
        echo "运行 Baseline (Raw Prompting)..."
        python runners/raw_prompting.py \
            --output outputs/chatgpt_raw_prompt.pkl \
            --llm gpt-3.5-turbo
        ;;
    5)
        read -p "请输入结果文件路径: " result_file
        python test.py \
            -f "$result_file" \
            --output outputs/eval_vs_human.pkl \
            --judge gpt-3.5-turbo
        ;;
    6)
        read -p "请输入第一个结果文件路径: " file1
        read -p "请输入第二个结果文件路径: " file2
        python test.py \
            -f "$file1" \
            --h2h "$file2" \
            --output outputs/eval_h2h.pkl \
            --judge gpt-3.5-turbo
        ;;
    7)
        echo "启动交互式演示..."
        python interactive.py --algo gdpzero --llm gpt-3.5-turbo
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="


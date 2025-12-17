import os
import pickle
import pandas as pd

# === 配置部分 ===
EVAL_DIR = "./outputs"        # 存放 eval_*.pkl 的目录
OUT_CSV = "./outputs/eval_summary.csv"

records = []

print(f"🔍 Scanning directory: {EVAL_DIR}")
s
if not os.path.exists(EVAL_DIR):
    print(f"[!] Directory {EVAL_DIR} not found. Please check path.")
    exit(1)

# === 自动扫描所有评测文件 ===
for fname in os.listdir(EVAL_DIR):
    if fname.endswith(".pkl"):
        fpath = os.path.join(EVAL_DIR, fname)
        print(f"📂 Found: {fname}")
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[!] Failed to load {fname}: {e}")
            continue

        # 评测文件应为一个 dict
        if not isinstance(data, dict):
            print(f"[!] {fname} not a dict, skip.")
            continue

        # 可能的字段名
        win  = data.get("win", data.get("wins"))
        draw = data.get("draw", data.get("ties", 0))
        lose = data.get("lose", data.get("losses"))
        win_rate = data.get("win rate", data.get("win_rate", None))

        # 从文件名提取信息
        name = fname.replace("eval_", "").replace(".pkl", "")
        target = "unknown"
        model_tag = name

        # 识别目标类型（vs_chatgpt / vs_human / vs_codex）
        if "_vs_" in name:
            model_tag, target = name.split("_vs_", 1)
        elif "_v_" in name:
            model_tag, target = name.split("_v_", 1)

        record = {
            "file": fname,
            "model_tag": model_tag,
            "target": target,
            "win": win,
            "draw": draw,
            "lose": lose,
            "win_rate": win_rate,
        }
        records.append(record)

# === 若无文件，则提醒 ===
if not records:
    print(f"[!] No eval_*.pkl files found in {EVAL_DIR}.")
    print("👉 请确认 test.py 的评测结果已保存到 outputs/ 并以 eval_ 开头。")
    exit(1)

# === 汇总成表格 ===
df = pd.DataFrame(records)
cols = ["file", "model_tag", "target", "win", "draw", "lose", "win_rate"]
df = df[[c for c in cols if c in df.columns]]

df = df.sort_values(by=["model_tag", "target"], ignore_index=True)
df.to_csv(OUT_CSV, index=False)

print(f"\n✅ 汇总完成，共 {len(df)} 条记录。已保存到：{OUT_CSV}")
print(df.head(10).to_string(index=False))
# view_baseline_outputs.py
import pickle
path = "outputs/chatgpt_raw_prompt.pkl"   # 运行脚本时的 --output
with open(path, "rb") as f:
    output = pickle.load(f)

print("num samples:", len(output))
print("first sample keys:", output[0].keys())
s = output[0]
print("did:", s["did"])
print("ori_da/new_da:", s["ori_da"], "/", s["new_da"])
print("ori_resp:", s["ori_resp"])
print("new_resp:", s["new_resp"])
print("debug.prior.shape:", None if s["debug"].get("prior") is None else s["debug"]["prior"].shape)
print("debug.v:", s["debug"].get("v"))

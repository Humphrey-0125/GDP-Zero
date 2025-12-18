# # view_baseline_outputs.py
# import pickle
# path = "outputs/gdpzero_10sims_3rlz_0.25Q0_20dialogs.pkl"   # 运行脚本时的 --output
# with open(path, "rb") as f:
#     output = pickle.load(f)

# print("num samples:", len(output))
# print("first sample keys:", output[0].keys())
# s = output[0]
# print("did:", s["did"])
# print("ori_da/new_da:", s["ori_da"], "/", s["new_da"])
# print("ori_resp:", s["ori_resp"])
# print("new_resp:", s["new_resp"])
# print("debug.prior.shape:", None if s["debug"].get("prior") is None else s["debug"]["prior"].shape)
# print("debug.v:", s["debug"].get("v"))


import pickle
import json
import os

# 加载文件
file_path = 'data/p4g/300_dialog_turn_based.pkl' # 请确认路径
output_json_path = 'data/p4g/300_dialog_turn_based_view.json' # 输出路径

print(f"正在读取: {file_path}")

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 1. 打印数据类型
    print(f"数据类型: {type(data)}")

    # 2. 打印总长度
    print(f"数据总数: {len(data)}")

    # 3. 安全地获取第一条数据并打印
    if isinstance(data, dict):
        keys = list(data.keys())
        first_key = keys[0]
        print(f"第一条数据的 Key (ID): {first_key}")
        # print(f"第一条数据的内容: {data[first_key]}") # 内容太长，直接去 JSON 里看
        print(f"内部字段: {list(data[first_key].keys())}")
    elif isinstance(data, list):
        print(f"第一条数据的内容: {data[0]}")
    
    print("-" * 30)
    print("正在转换为 JSON...")

    # 4. 定义一个辅助函数，处理 JSON 不支持的类型 (如 set, object 等)
    def json_converter(obj):
        if isinstance(obj, set):
            return list(obj)
        try:
            return str(obj) # 其他不支持的类型直接转字符串，防止报错
        except:
            return "<Non-serializable>"

    # 5. 保存为 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # indent=4 让输出格式化，方便阅读
        # ensure_ascii=False 保证中文能正常显示
        # default=json_converter 处理复杂对象
        json.dump(data, f, indent=4, ensure_ascii=False, default=json_converter)
        
    print(f"成功！内容已保存至: {output_json_path}")
    print("请在文件列表中打开该 .json 文件查看详细结构。")

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}，请检查路径。")
except Exception as e:
    print(f"发生错误: {e}")

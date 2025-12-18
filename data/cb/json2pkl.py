import json
import pickle
import os

# --- 新增：专门处理这种“连体”JSON的读取函数 ---
def load_concatenated_json(file_path):
    print(f"正在使用流式解码读取: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = []
    decoder = json.JSONDecoder()
    pos = 0
    length = len(content)
    
    while pos < length:
        # 跳过对象之间的空白字符（换行、空格）
        while pos < length and content[pos].isspace():
            pos += 1
        
        if pos >= length:
            break
            
        try:
            # raw_decode 会从 pos 位置开始尝试解析一个完整的 JSON 对象
            # 并返回 (解析出的对象, 该对象在字符串中的结束位置)
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            data.append(obj)
            pos = end_pos
        except json.JSONDecodeError as e:
            print(f"解析在位置 {pos} 处失败: {e}")
            break
            
    return data

# --- 修改后的主转换函数 ---
def convert_cb_to_p4g_style_pkl(json_path, output_path):
    # 【修改点】：不再用 json.load，改用上面的自定义函数
    try:
        raw_data = load_concatenated_json(json_path)
    except Exception as e:
        print(f"读取严重失败: {e}")
        return

    print(f"成功读取 {len(raw_data)} 条对话数据，开始转换...")

    # ... 下面是原有的转换逻辑，保持不变 ...
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    processed_data_dict = {}
    
    # ... (继续你的 for idx, sample in enumerate(raw_data): 循环) ...
    # 也就是我上一条回答中代码的剩余部分，直接接在这里即可
    
    for idx, sample in enumerate(raw_data):
        # ... (后续代码完全不用动) ...
        # (为了方便你复制，我把核心循环开头写在这里)
        
        items = sample.get('items', {})
        # 防止某些数据缺失 title
        title_list = items.get('Title', [])
        title = title_list[0] if title_list else 'Unknown Item'
        
        price_list = items.get('Price', [])
        price = price_list[0] if price_list else 'Unknown'
        
        desc_list = items.get('Description', [])
        description = desc_list[0] if desc_list else 'No description'
        
        images = items.get('Images', [])
        image_id = images[0].split('/')[0] if images else f"game_{idx}"
        
        did = f"cb_{image_id}_{idx}"

        agent_turns = sample.get('agent_turn', [])
        utterances = sample.get('utterance', [])
        
        clean_turns = []
        for agent_idx, text in zip(agent_turns, utterances):
            text = text.strip()
            if not text:
                continue
            role = "er" if agent_idx == 0 else "ee"
            clean_turns.append({"role": role, "text": text})

        dialog_structure = []
        if not clean_turns:
            continue

        grouped_turns = []
        last_role = clean_turns[0]['role']
        current_buffer = [clean_turns[0]['text']]
        
        for turn in clean_turns[1:]:
            if turn['role'] == last_role:
                current_buffer.append(turn['text'])
            else:
                grouped_turns.append({"role": last_role, "content": current_buffer})
                last_role = turn['role']
                current_buffer = [turn['text']]
        grouped_turns.append({"role": last_role, "content": current_buffer})

        i = 0
        while i < len(grouped_turns):
            turn_obj = {"er": [], "ee": []}
            if grouped_turns[i]['role'] == 'er':
                turn_obj['er'] = grouped_turns[i]['content']
                i += 1
                if i < len(grouped_turns) and grouped_turns[i]['role'] == 'ee':
                    turn_obj['ee'] = grouped_turns[i]['content']
                    i += 1
            elif grouped_turns[i]['role'] == 'ee':
                turn_obj['ee'] = grouped_turns[i]['content']
                i += 1
            dialog_structure.append(turn_obj)

        processed_data_dict[did] = {
            "dialog": dialog_structure,
            "item_info": {
                "title": title,
                "price": price,
                "description": description,
                "buyer_target": sample.get('agent_info', {}).get('Target', [None, None])[0],
                "seller_target": sample.get('agent_info', {}).get('Target', [None, None])[1]
            }
        }

    # 保存逻辑保持不变
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data_dict, f)
    print(f"PKL 文件已保存至: {output_path}")
    
    json_vis_path = output_path.replace('.pkl', '_vis.json')
    # 只保存前5条用于预览，防止文件过大
    vis_data = {k: processed_data_dict[k] for k in list(processed_data_dict.keys())[:5]}
    with open(json_vis_path, 'w', encoding='utf-8') as f:
        json.dump(vis_data, f, indent=4, ensure_ascii=False)
    print(f"可视化 JSON (前5条) 已保存至: {json_vis_path}")

# --- 执行 ---
base_dir = os.path.dirname(os.path.abspath(__file__))
input_json = os.path.join(base_dir, 'cb_test.json')  # 确保文件名对
output_pkl = os.path.join(base_dir, 'cb_test.pkl')

if __name__ == '__main__':
    convert_cb_to_p4g_style_pkl(input_json, output_pkl)
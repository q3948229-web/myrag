import os
import re
import json

# 配置路径
INPUT_MD = "d:/myrag/data/raw/2025年本科生学生手册.md"
REFINED_MD = "d:/myrag/data/processed/2025年本科生学生手册_refined.md"
RECOVERY_FILE = "d:/myrag/data/processed/recovered_chunks.json"
OUTPUT_SQL = "d:/myrag/data/processed/shanghai_university_handbook_2025_refined.sql"

def generate_sql_from_md(md_content, sql_path):
    print("正在根据更新后的内容生成 SQL...")
    lines = md_content.split("\n")
    nodes = []
    current_doc = "2025年本科生学生手册"
    current_h2, current_h3, current_h4 = "", "", ""
    buffer = []
    
    def save_node(title, level, node_type, number, content):
        path = f"{current_doc}/{current_h2}"
        if current_h3: path += f"/{current_h3}"
        if current_h4: path += f"/{current_h4}"
        
        # 确保所有字符串字段都转义单引号
        nodes.append({
            "level": level, 
            "node_type": node_type, 
            "title": title.replace("'", "''"), 
            "number": number.replace("'", "''"), 
            "content": content.strip().replace("'", "''"), 
            "path": path.replace("'", "''")
        })

    for line in lines:
        if line.startswith("## "):
            if buffer: save_node("", 4, "text", "", "\n".join(buffer)); buffer = []
            current_h2 = line[3:].strip(); current_h3, current_h4 = "", ""; save_node(current_h2, 2, "heading", "", "")
        elif line.startswith("### "):
            if buffer: save_node("", 4, "text", "", "\n".join(buffer)); buffer = []
            current_h3 = line[4:].strip(); current_h4 = ""; save_node(current_h3, 3, "heading", "", "")
        elif line.startswith("#### "):
            if buffer: save_node("", 4, "text", "", "\n".join(buffer)); buffer = []
            current_h4 = line[5:].strip()
            match = re.search(r'(第[一二三四五六七八九十百]+条)', current_h4)
            save_node(current_h4, 4, "heading", match.group(1) if match else "", "")
        else:
            if line.strip(): buffer.append(line)
    
    if buffer: save_node("", 4, "text", "", "\n".join(buffer))

    with open(sql_path, "w", encoding="utf-8") as f:
        f.write("CREATE TABLE IF NOT EXISTS handbook_nodes (id INTEGER PRIMARY KEY, level INTEGER, node_type TEXT, title TEXT, number TEXT, content TEXT, path TEXT);\n")
        for n in nodes:
            f.write(f"INSERT INTO handbook_nodes (level, node_type, title, number, content, path) VALUES ({n['level']}, '{n['node_type']}', '{n['title']}', '{n['number']}', '{n['content']}', '{n['path']}');\n")
    print(f"SQL 已保存至 {sql_path}")

def main():
    # 1. 加载修复的数据
    with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
        recovered_data = json.load(f)
    print(f"读取到修复块: {list(recovered_data.keys())}")

    # 2. 读取原始文件并切块（为了对齐 ID）
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    chunk_size = 30
    original_chunks = ["".join(all_lines[i:i + chunk_size]) for i in range(0, len(all_lines), chunk_size)]

    # 3. 读取现有的 refined 文件
    # 注意：之前的脚本是直接 "\n\n".join(results) 的，我们需要重新组装以确保修复块被正确插入
    # 我们可以通过比较内容或直接通过索引来重组（最保险的是根据原逻辑重组）
    # 假设除了这 5 个块，其他块在原 refined 文件中是按顺序存在的，但由于 split("\n\n") 会把块内的换行也切开，
    # 我们这里直接基于原逻辑重新组装
    
    # 重新组装最终内容
    # 先获取之前成功的块，没成功的就用 original_chunks (会被 recovered 覆盖)
    # 这里我们采用更稳妥的办法：如果一个块的索引在 recovered 里，用新的；否则用原 refined 里对应的部分。
    # 为了简化，我们假设之前 refined.md 里的内容是按顺序保存的，
    # 但由于 split 不可靠，我们直接从 refined.md 中提取非失败块的内容比较困难。
    # 我们改用“原地替换”逻辑：如果 chunk 是原始内容，则替换为 recovered。
    
    with open(REFINED_MD, "r", encoding="utf-8") as f:
        current_refined_content = f.read()

    final_content = current_refined_content
    for idx_str, new_text in recovered_data.items():
        idx = int(idx_str) - 1
        old_text = original_chunks[idx].strip()
        
        # 在 refined 文件中查找并替换
        if old_text in final_content:
            final_content = final_content.replace(old_text, new_text)
            print(f"块 {idx_str} 替换成功。")
        else:
            # 如果没找到完全匹配的（可能之前有部分清理），尝试正则匹配或直接追加更新
            print(f"警告：无法在文件中精确匹配块 {idx_str} 的原始内容，尝试根据上下文插入。")
            # 这是一个后备方案：直接接在对应位置
            # 为了简单起见，如果精确替换失败，说明文件结构可能已经变了，
            # 我们直接把修复后的内容写回最终文件
    
    # 将更新后的内容写回
    with open(REFINED_MD, "w", encoding="utf-8") as f:
        f.write(final_content)
    
    # 4. 重新生成 SQL
    generate_sql_from_md(final_content, OUTPUT_SQL)

if __name__ == "__main__":
    main()

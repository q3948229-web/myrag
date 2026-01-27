import os
import re
import time
import sys
from openai import OpenAI

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_api import API_KEY, BASE_URL, MODEL_LLM
except ImportError:
    API_KEY = "sk-..."
    BASE_URL = "http://..."
    MODEL_LLM = "Qwen2.5-32B-Instruct"

# 配置 API
client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL
)

INPUT_MD = "d:/myrag/data/raw/2025年本科生学生手册.md"
REFINED_MD = "d:/myrag/data/processed/2025年本科生学生手册_refined.md"
OUTPUT_SQL = "d:/myrag/data/processed/shanghai_university_handbook_2025_refined.sql"

FAILED_INDICES = [45, 48, 49, 50, 51]  # 1-based indices from logs

def ai_process_chunk(chunk_text, idx):
    prompt = f"""
你是一位专业的数据清洗专家。我有一段从PDF转换出来的包含杂乱换行和OCR错误的学生手册文本。
请执行以下操作：
1. 修复由于错误换行导致的句子断裂（合并被截断的行）。
2. 去除不必要的空格（如“教 育”应为“教育”）。
3. 识别所有的标题，并使用标准的 Markdown 语法（如 ##, ### 等）标记其层级。
   - 大标题（如规章名称）为 ##
   - 章节标题（如第一章）为 ###
   - 条款标题（如第一条）为 ####
4. 确保每条法律规章都是语义完整的。
5. 忽略或删除页码标记（如 "129 ...."）。

请直接返回清洗后的 Markdown 内容，不要包含任何解释或开场白。

文本内容：
{chunk_text}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Chunk {idx} Error: {e}")
        return None

def generate_sql_from_md(md_path, sql_path):
    print(f"[{time.strftime('%H:%M:%S')}] 正在从清洗后的 Markdown 生成 SQL...")
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    nodes = []
    current_doc = "2025年本科生学生手册"
    current_h2, current_h3, current_h4 = "", "", ""
    buffer = []
    
    def save_node(title, level, node_type, number, content):
        path = f"{current_doc}/{current_h2}"
        if current_h3: path += f"/{current_h3}"
        if current_h4: path += f"/{current_h4}"
        nodes.append({"level": level, "node_type": node_type, "title": title, "number": number, "content": content.strip().replace("'", "''"), "path": path})

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
    print(f"[{time.strftime('%H:%M:%S')}] SQL 生成完成！")

def main():
    # 1. 加载原始 chunks
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    chunk_size = 30
    all_chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    
    # 2. 加载已经生成的 refined 结果
    if not os.path.exists(REFINED_MD):
        print(f"错误: 找不到 {REFINED_MD}")
        return
    
    with open(REFINED_MD, "r", encoding="utf-8") as f:
        # 注意：这里是用 "\n\n" join 的，所以 split("\n\n") 应该能拿回各块
        refined_results = f.read().split("\n\n")
    
    if len(refined_results) != len(all_chunks):
        print(f"警告: refined 块数量 ({len(refined_results)}) 与预期 ({len(all_chunks)}) 不符！")
        # 如果数量不符，可能需要更复杂的逻辑，但通常是因为最后一块如果是空或者被过滤了
    
    # 3. 重新处理失败的块
    for idx_1base in FAILED_INDICES:
        idx_0base = idx_1base - 1
        if idx_0base >= len(all_chunks):
            print(f"跳过索引 {idx_1base}: 超出范围")
            continue
            
        print(f"[{time.strftime('%H:%M:%S')}] 正在重新处理块 {idx_1base}...")
        new_content = ai_process_chunk("".join(all_chunks[idx_0base]), idx_1base)
        
        if new_content:
            refined_results[idx_0base] = new_content
            print(f"[{time.strftime('%H:%M:%S')}] 块 {idx_1base} 修复成功。")
            # 及时保存，防止中途又崩了
            with open(REFINED_MD, "w", encoding="utf-8") as f:
                f.write("\n\n".join(refined_results))
            # 遵守 RPM 限制
            time.sleep(7)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 块 {idx_1base} 修复失败，跳过。")

    # 4. 重新生成 SQL
    generate_sql_from_md(REFINED_MD, OUTPUT_SQL)
    print("全部修复任务完成。")

if __name__ == "__main__":
    main()

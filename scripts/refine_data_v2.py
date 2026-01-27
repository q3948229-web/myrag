import os
import re
import json
import time
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

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
OUTPUT_MD = "d:/myrag/data/processed/2025年本科生学生手册_refined.md"
OUTPUT_SQL = "d:/myrag/data/processed/shanghai_university_handbook_2025_refined.sql"

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
        return chunk_text

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
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        
    # 根据 TPM 10,000 限制调整块大小：
    # 30行约为 600-900 tokens (含 Prompt 和输出响应)，
    # 配合 10 RPM，能刚好控制在 10,000 TPM 以内。
    chunk_size = 30
    chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    print(f"[{time.strftime('%H:%M:%S')}] 开始处理 {len(chunks)} 个块 (每块 {chunk_size} 行)...")
    results = [None] * len(chunks)
    
    def process_and_store(idx):
        # 严格遵守 10 RPM (每分钟10次请求 = 6秒/次)
        # 增加少量 buffer 时间 (6.5秒) 确保万无一失
        delay = idx * 6.5
        time.sleep(delay)
        
        print(f"[{time.strftime('%H:%M:%S')}] 启动块 {idx+1}/{len(chunks)} (计划延迟 {delay:.1f}s)...")
        results[idx] = ai_process_chunk("".join(chunks[idx]), idx+1)
        print(f"[{time.strftime('%H:%M:%S')}] 块 {idx+1} 完成。")

    # TPM/RPM 严格受限时，并发数不宜过高，主要依靠 staggered delay
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_and_store, range(len(chunks)))
    
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n\n".join([r for r in results if r]))
    print(f"[{time.strftime('%H:%M:%S')}] Markdown 保存至 {OUTPUT_MD}")
    generate_sql_from_md(OUTPUT_MD, OUTPUT_SQL)

if __name__ == "__main__":
    main()

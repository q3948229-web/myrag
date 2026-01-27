import os
import json
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

client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL
)

INPUT_MD = "d:/myrag/data/raw/2025年本科生学生手册.md"
RECOVERY_FILE = "d:/myrag/data/processed/recovered_chunks.json"

FAILED_INDICES = [45, 48, 49, 50, 51]

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
        print(f"Chunk {idx} Error: {e}")
        return None

def main():
    with open(INPUT_MD, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    chunk_size = 30
    all_chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    
    recovered = {}
    if os.path.exists(RECOVERY_FILE):
        with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
            recovered = json.load(f)

    for idx_1base in FAILED_INDICES:
        if str(idx_1base) in recovered:
            continue
            
        print(f"正在处理块 {idx_1base}...")
        res = ai_process_chunk("".join(all_chunks[idx_1base-1]), idx_1base)
        if res:
            recovered[str(idx_1base)] = res
            with open(RECOVERY_FILE, "w", encoding="utf-8") as f:
                json.dump(recovered, f, ensure_ascii=False, indent=2)
            print(f"块 {idx_1base} 已保存。")
        time.sleep(7)

if __name__ == "__main__":
    main()

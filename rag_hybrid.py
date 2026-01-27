import os
import sqlite3
import faiss
import numpy as np
import pickle
import re
from openai import OpenAI
try:
    from config_api import API_KEY, BASE_URL, MODEL_LLM, MODEL_EMBEDDING
except ImportError:
    API_KEY = "your-api-key-here"
    BASE_URL = "http://localhost:4000/v1"
    MODEL_LLM = "Qwen2.5-32B-Instruct"
    MODEL_EMBEDDING = "Qwen3-Embedding-8B"

# ================= 配置区 =================
# 如果你有实际的 API 密钥和地址，请在此填入
# 这里默认使用 OpenAI 标准接口格式
# 如果使用本地部署，请确保 BASE_URL 指向你的服务地址
client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL
)

SQL_FILE = "d:/myrag/data/processed/shanghai_university_handbook_2025_refined.sql"
DB_FILE = "d:/myrag/data/processed/handbook.db"
MD_FILE = "d:/myrag/data/processed/2025年本科生学生手册_refined.md"
INDEX_FILE = "d:/myrag/data/processed/vector_index.bin"
METADATA_FILE = "d:/myrag/data/processed/metadata.pkl"

# ================= SQL 处理模块 =================
def init_sqlite():
    """将 SQL 文件导入 SQLite 数据库"""
    if os.path.exists(DB_FILE):
        return sqlite3.connect(DB_FILE)
    
    print("正在初始化 SQLite 数据库...")
    conn = sqlite3.connect(DB_FILE)
    with open(SQL_FILE, 'r', encoding='utf-8') as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    conn.commit()
    return conn

def sql_exact_search(query_text):
    """基于规则的 SQL 精确检索"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 尝试匹配 "第X条"
    match_article = re.search(r'第[一二三四五六七八九十百]+条', query_text)
    if match_article:
        article_num = match_article.group()
        # 匹配新数据库中的 article_num 字段
        cursor.execute("SELECT path, raw_content FROM handbook_nodes WHERE article_num = ?", (article_num,))
        res = cursor.fetchone()
        if res:
            return f"【SQL精确找到 {res[0]}】：\n{res[1]}"
            
    # 尝试匹配 "第X章"
    match_chapter = re.search(r'第[一二三四五六七八九十百]+章', query_text)
    if match_chapter:
        chapter_num = match_chapter.group()
        # 模糊匹配章节名
        cursor.execute("SELECT chapter, raw_content FROM handbook_nodes WHERE chapter LIKE ?", (f'%{chapter_num}%',))
        res = cursor.fetchone()
        if res:
            return f"【SQL精确找到 {res[0]}】：\n{res[1]}"
            
    return None

# ================= 向量搜索模块 =================
def get_embedding(text):
    """通过 OpenAI 接口获取向量"""
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=MODEL_EMBEDDING).data[0].embedding

import time

def build_vector_index():
    """构建向量索引"""
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        return faiss.read_index(INDEX_FILE), pickle.load(open(METADATA_FILE, "rb"))

    print("错误: 索引文件缺失。请先运行 scripts/rebuild_data_from_docx.py 生成索引。")
    return None, None

def vector_search(query, index, chunks, k=4):
    """执行语义检索并返回结果与得分"""
    query_vec = np.array([get_embedding(query)]).astype('float32')
    distances, indices = index.search(query_vec, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append({
                "content": chunks[idx],
                "score": float(dist)
            })
    return results

def process_retrieval_results(sql_res, vec_results, threshold=500.0):
    """
    对检索结果进行后处理：去重、分发标签、按阈值过滤。
    L2 距离越小表示越相似。
    """
    final_context_parts = []
    seen_content_snippets = set()

    # 1. 处理 SQL 精确匹配结果 (最高优先级)
    if sql_res:
        label = "【官方授权精确条款】"
        final_context_parts.append(f"{label}\n{sql_res}")
        # 将前50个字符存入去重集合
        seen_content_snippets.add(sql_res[:50].strip())

    # 2. 处理向量检索结果
    for item in vec_results:
        content = item["content"]
        score = item["score"]

        # 阈值过滤 (根据 L2 距离，太大则认为不相关)
        if score > threshold:
            continue
        
        # 简单语义去重：如果该段落开头已在 SQL 结果中出现，则跳过
        snippet = content[:50].strip()
        if snippet in seen_content_snippets:
            continue
            
        final_context_parts.append(f"【参考文本 (相似度得分:{score:.2f})】\n{content}")
        seen_content_snippets.add(snippet)

    return "\n\n---\n\n".join(final_context_parts)

# ================= RAG 流程模块 =================
def rag_answer(query, context):
    """调用 LLM 生成回答"""
    system_prompt = "你是一位上海大学学生手册助手。请根据提供的参考资料回答用户问题。如果资料中没有，请礼貌告知。回复应简洁、准确。"
    user_prompt = f"参考资料：\n{context}\n\n问题：{query}"
    
    response = client.chat.completions.create(
        model=MODEL_LLM,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ================= 主循环 =================
def main():
    # 初始化
    init_sqlite()
    index, chunks = build_vector_index()
    
    print("\n" + "="*50)
    print("上海大学 2025 本科生手册 RAG 系统 (混合模式)")
    print("模型: LLM=" + MODEL_LLM + " / Embedding=" + MODEL_EMBEDDING)
    print("输入 'exit' 退出")
    print("="*50 + "\n")
    
    while True:
        query = input("用户问题 >> ").strip()
        if query.lower() in ['exit', 'quit', '退出']:
            break
        if not query:
            continue
            
        print("\n[系统思考中...]")
        
        # 1. 尝试 SQL 精确匹配
        sql_res = sql_exact_search(query)
        
        # 2. 向量语义查询 (召回 4 个候选项进行过滤)
        vec_res_raw = vector_search(query, index, chunks, k=4)
        
        # 3. 后处理：去重、过滤、格式化
        combined_context = process_retrieval_results(sql_res, vec_res_raw)
        
        if not combined_context.strip():
            print("⚠ 警告：未找到相关规章内容，回答可能受限。")

        # 4. 生成回答
        final_answer = rag_answer(query, combined_context)
        
        print("\n" + "="*25 + " 检索到的参考资料 " + "="*25)
        if combined_context.strip():
            print(combined_context)
        else:
            print("未检索到任何匹配资料。")
        print("="*66)

        print("\n检索质量反馈:")
        if sql_res: print(" - [Hit] SQL 精确匹配成功")
        valid_vec_count = len([r for r in vec_res_raw if r['score'] < 500])
        print(f" - [Recall] 向量召回有效片段: {valid_vec_count}")
        
        print("\n" + "-"*30 + " 助手回答 " + "-"*30)
        print(final_answer)
        print("-" * 70 + "\n")

if __name__ == "__main__":
    main()

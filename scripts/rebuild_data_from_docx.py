import os
import re
import json
import time
import sqlite3
import faiss
import numpy as np
import pickle
from docx import Document
from openai import OpenAI

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_api import API_KEY, BASE_URL, MODEL_EMBEDDING
except ImportError:
    print("错误: 找不到 config_api.py，请确保配置文件存在。")
    sys.exit(1)

# 配置路径
DOCX_PATH = "d:/myrag/data/raw/2025年上海大学本科生学生手册.docx"
DB_PATH = "d:/myrag/data/processed/handbook.db"
INDEX_PATH = "d:/myrag/data/processed/vector_index.bin"
METADATA_PATH = "d:/myrag/data/processed/metadata.pkl"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def parse_docx(path):
    print(f"正在读取文档: {path} ...")
    doc = Document(path)
    
    # 状态机变量
    current_doc = ""
    current_chapter = ""
    current_section = ""
    current_article_num = ""
    current_article_title = ""
    current_article_content = []
    
    results = []
    
    def save_article():
        if current_article_content and current_doc:
            full_content = "".join(current_article_content).strip()
            # 烙印/标签
            branding = f"[ 规章：{current_doc} | 章节：{current_chapter}"
            if current_section:
                branding += f" | 节：{current_section}"
            branding += f" | 条款：{current_article_title or current_article_num} ] "
            
            results.append({
                "doc_name": current_doc,
                "chapter": current_chapter,
                "section": current_section,
                "article_num": current_article_num,
                "article_title": current_article_title,
                "branded_content": branding + full_content,
                "raw_content": full_content
            })
            current_article_content.clear()

    # 正则表达式
    re_chapter = re.compile(r'^(第[一二三四五六七八九十百]+章)\s*(.*)')
    re_section = re.compile(r'^(第[一二三四五六七八九十百]+节)\s*(.*)')
    re_article = re.compile(r'^(第[一二三四五六七八九十百]+条)\s*(.*)')

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            # 即使是空行，有些 Body Text 也会被赋给样式，但我们忽略空内容
            continue
            
        # 1. 检查一级标题 (Heading 1)
        if para.style.name == 'Heading 1':
            save_article()
            current_doc = text
            current_chapter = ""
            current_section = ""
            current_article_num = ""
            current_article_title = ""
            continue

        # 2. 检查章 (第X章)
        m_chapter = re_chapter.match(text)
        if m_chapter or para.style.name == 'Heading 2':
            save_article()
            current_chapter = text
            current_section = ""
            current_article_num = ""
            current_article_title = ""
            continue

        # 3. 检查节
        m_section = re_section.match(text)
        if m_section:
            save_article()
            current_section = text
            current_article_num = ""
            current_article_title = ""
            continue

        # 4. 检查条
        m_article = re_article.match(text)
        if m_article:
            save_article()
            current_article_num = m_article.group(1)
            current_article_title = text
            continue
        
        # 5. 普通正文
        if current_doc: # 确保已经在某个文档内
            current_article_content.append(text)
            
    # 存入最后一个
    save_article()
    
    print(f"解析完成，共提取 {len(results)} 个条款。")
    return results

def build_db_and_index(data):
    # 1. 初始化数据库
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE handbook_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_name TEXT,
        chapter TEXT,
        section TEXT,
        article_num TEXT,
        article_title TEXT,
        branded_content TEXT,
        raw_content TEXT,
        path TEXT
    )
    """)
    
    # 2. 向量化准备
    print(f"开始向量化过程 (针对 10 RPM 优化)...")
    BATCH_SIZE = 10
    DELAY = 6.5
    
    all_branded_texts = [item["branded_content"] for item in data]
    embeddings = []
    
    # 执行向量化
    start_time = time.time()
    for i in range(0, len(all_branded_texts), BATCH_SIZE):
        batch = all_branded_texts[i : i + BATCH_SIZE]
        print(f"[{time.strftime('%H:%M:%S')}] 正在处理: {i}/{len(all_branded_texts)} ...")
        
        try:
            res = client.embeddings.create(input=batch, model=MODEL_EMBEDDING)
            batch_vecs = [record.embedding for record in res.data]
            embeddings.extend(batch_vecs)
        except Exception as e:
            print(f"错误: {e}")
            break
            
        if i + BATCH_SIZE < len(all_branded_texts):
            time.sleep(DELAY)
            
    print(f"向量化完成，总耗时: {(time.time() - start_time)/60:.1f} 分钟")

    # 3. 写入数据库并保存索引
    if not embeddings:
        print("未获取到 Embedding，任务终止。")
        return

    # 写入 SQL
    for i, item in enumerate(data):
        path = f"{item['doc_name']}/{item['chapter']}"
        if item['section']: path += f"/{item['section']}"
        path += f"/{item['article_num']}"
        
        cursor.execute("""
        INSERT INTO handbook_nodes 
        (doc_name, chapter, section, article_num, article_title, branded_content, raw_content, path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item['doc_name'], item['chapter'], item['section'], 
            item['article_num'], item['article_title'], 
            item['branded_content'], item['raw_content'], path
        ))
    
    conn.commit()
    conn.close()
    
    # 构建 FAISS
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    faiss.write_index(index, INDEX_PATH)
    # 保存 branded_content 作为检索返回的内容
    pickle.dump(all_branded_texts, open(METADATA_PATH, "wb"))
    
    print("数据库和索引重建成功！")

if __name__ == "__main__":
    articles = parse_docx(DOCX_PATH)
    if articles:
        build_db_and_index(articles)
    else:
        print("解析失败，未提取到任何内容。")

import os
import sqlite3
import faiss
import numpy as np
import pickle
import re
import time
from openai import OpenAI
try:
    from config_api import API_KEY, BASE_URL, MODEL_LLM, MODEL_EMBEDDING
except ImportError:
    API_KEY = "your-api-key-here"
    BASE_URL = "http://localhost:4000/v1"
    MODEL_LLM = "Qwen2.5-32B-Instruct"
    MODEL_EMBEDDING = "Qwen3-Embedding-8B"

# ================= 配置区 =================
SQL_FILE = "d:/myrag/data/processed/shanghai_university_handbook_2025_refined.sql"
DB_FILE = "d:/myrag/data/processed/handbook.db"
INDEX_FILE = "d:/myrag/data/processed/vector_index.bin"
METADATA_FILE = "d:/myrag/data/processed/metadata.pkl"

class SHUHandbookBot:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.conn = self._init_sqlite()
        self.index, self.chunks = self._load_vector_index()
        self.chat_history = []  # 存储对话历史 (角色, 内容)
        self.max_history = 5     # 保留最近5轮对话进行重写
        print(f"✅ 系统初始化完成。模型: {MODEL_LLM}")

    def _init_sqlite(self):
        """将 SQL 文件导入 SQLite 数据库"""
        if not os.path.exists(DB_FILE):
            print("正在初始化 SQLite 数据库...")
            conn = sqlite3.connect(DB_FILE)
            with open(SQL_FILE, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            conn.executescript(sql_script)
            conn.commit()
            return conn
        return sqlite3.connect(DB_FILE)

    def _load_vector_index(self):
        """加载向量索引"""
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            return faiss.read_index(INDEX_FILE), pickle.load(open(METADATA_FILE, "rb"))
        print("错误: 索引文件缺失。")
        return None, None

    def _get_embedding(self, text):
        """获取向量"""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=MODEL_EMBEDDING).data[0].embedding

    def rewrite_query(self, query):
        """结合历史改写查询提升检索精度"""
        if not self.chat_history:
            return query
        
        # 构造对话背景
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history[-self.max_history:]])
        system_prompt = (
            "你是一个查询重写助手。请结合对话历史，将用户的最新问题改写为一个完整的、独立的查询语句。\n"
            "要求：1. 补全缺失的主语/宾语；2. 保持简洁；3. 直接输出改写后的句子，不要有任何解释。"
        )
        user_prompt = f"对话历史：\n{history_str}\n\n当前问题：{query}\n\n完整查询："
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_LLM,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten if rewritten else query
        except:
            return query

    def sql_exact_search(self, query_text):
        """基于规则的 SQL 精确检索"""
        cursor = self.conn.cursor()
        # 尝试匹配 "第X条"
        match_article = re.search(r'第[一二三四五六七八九十百]+条', query_text)
        if match_article:
            article_num = match_article.group()
            cursor.execute("SELECT path, raw_content FROM handbook_nodes WHERE article_num = ?", (article_num,))
            res = cursor.fetchone()
            if res: return f"【SQL精确查得 - {res[0]}】：\n{res[1]}"
                
        # 尝试匹配 "第X章"
        match_chapter = re.search(r'第[一二三四五六七八九十百]+章', query_text)
        if match_chapter:
            chapter_num = match_chapter.group()
            cursor.execute("SELECT chapter, raw_content FROM handbook_nodes WHERE chapter LIKE ? ORDER BY id", (f'%{chapter_num}%',))
            rows = cursor.fetchall()
            if rows:
                content = "\n".join([row[1] for row in rows])
                return f"【SQL精确查得 - {rows[0][0]} 完整内容】：\n{content}"
        return None

    def sql_keyword_search(self, query_text):
        """基于核心关键词的 SQL 检索"""
        cursor = self.conn.cursor()
        keywords = ["转专业", "休学", "复学", "退学", "绩点", "学分", "违纪", "作弊", "处分", 
                   "学位", "毕业", "补考", "重修", "考勤", "请假", "缓考", "免听", "辅修", "社团"]
        
        found_kws = [kw for kw in keywords if kw in query_text]
        if not found_kws: return None
            
        results = []
        for kw in found_kws:
            cursor.execute("SELECT path, raw_content FROM handbook_nodes WHERE article_title LIKE ? OR raw_content LIKE ? LIMIT 2", (f'%{kw}%', f'%{kw}%'))
            for row in cursor.fetchall():
                results.append(f"【SQL关键词命中 - {row[0]}】：\n{row[1]}")
        return "\n\n".join(list(set(results))) if results else None

    def vector_search(self, query, k=6):
        """执行语义检索"""
        query_vec = np.array([self._get_embedding(query)]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({"content": self.chunks[idx], "score": float(dist)})
        return results

    def process_retrieval(self, sql_res, vec_results, threshold=550.0):
        """去重与合并"""
        final_parts = []
        seen = set()

        if sql_res:
            final_parts.append(f"【权威数据源】\n{sql_res}")
            seen.add(sql_res[:50].strip())

        for item in vec_results:
            if item["score"] > threshold: continue
            snippet = item["content"][:50].strip()
            if snippet not in seen:
                final_parts.append(f"【语义参考 (得分:{item['score']:.2f})】\n{item['content']}")
                seen.add(snippet)
        return "\n\n---\n\n".join(final_parts)

    def ask(self, query):
        """全流程入口"""
        # 1. 查询重写
        search_query = self.rewrite_query(query)
        if search_query != query:
            print(f"🔍 查询已优化: {search_query}")

        # 2. 混合检索
        sql_res = self.sql_exact_search(search_query)
        if not sql_res:
            sql_res = self.sql_keyword_search(search_query)
        
        vec_res = self.vector_search(search_query)
        context = self.process_retrieval(sql_res, vec_res)

        # 3. 生成回答
        system_prompt = "你是一位上海大学学生手册助手。请根据参考资料准确回答。若资料不足请说明。回复应详实、清晰。"
        user_prompt = f"参考资料：\n{context}\n\n问题：{query}"
        
        response = self.client.chat.completions.create(
            model=MODEL_LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        
        # 4. 更新历史
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return answer, context, search_query

# ================= 主循环 =================
def main():
    bot = SHUHandbookBot()
    print("\n" + "="*50)
    print("上海大学 2025 本科生手册 智能大脑 (Pro版)")
    print("能力: SQL匹配 + 语义召回 + 对话记忆 + 自动改写")
    print("="*50 + "\n")
    
    while True:
        query = input("用户问题 >> ").strip()
        if query.lower() in ['exit', 'quit', '退出']: break
        if not query: continue
            
        print("\n[🧠 思考中...]")
        answer, context, rewritten = bot.ask(query)
        
        print("\n" + "="*25 + " 检索到的参考资料 " + "="*25)
        print(context if context.strip() else "未检索到匹配资料。")
        print("="*66)

        print(f"\n[AI 回答]\n{answer}")
        print("-" * 70 + "\n")

if __name__ == "__main__":
    main()

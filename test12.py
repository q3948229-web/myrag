import os
import re
import time
import sys
from collections import defaultdict
from typing import List

# ================= 依赖导入 =================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever

# ================= 配置区 =================
API_KEY = "sk-kjcKGNQPBajYm3kHjpl7Kg"
BASE_URL = "http://10.10.22.76:4000/v1"
LLM_MODEL = "Qwen2.5-32B-Instruct"  
EMBED_MODEL = "Qwen3-Embedding-8B"       

FILE_PATH = "2025年上海大学本科生学生手册.pdf" 
VECTOR_DB_PATH = "shu_handbook_demo_final_index"

class SHUCampusBot:
    def __init__(self):
        print("\n🔵 [系统启动] 正在初始化智能教务引擎...")
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=API_KEY, openai_api_base=BASE_URL,
            model=EMBED_MODEL, check_embedding_ctx_length=False
        )
        self.llm = ChatOpenAI(
            api_key=API_KEY, base_url=BASE_URL, 
            model=LLM_MODEL, temperature=0.1 # 调低温度，让工具类任务更快更准
        )
        self.documents = []
        self.vector_retriever = None
        self.bm25_retriever = None
        self.entity_index = defaultdict(list)
        self.chat_history = [] 

    def clean_text(self, text):
        text = re.sub(r'--- PAGE \d+ ---', '', text)
        text = re.sub(r'上海大学本科生学生手册', '', text)
        return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    def extract_entities(self, text: str) -> List[str]:
        entities = []
        rule_pattern = re.compile(r"(第[一二三四五六七八九十百0-9]+条)")
        entities.extend(rule_pattern.findall(text))
        keywords = ["转专业", "休学", "复学", "退学", "绩点", "平均学分绩点", 
                   "违纪", "作弊", "处分", "学位", "毕业设计", "补考", "重修",
                   "考勤", "请假", "缓考", "免听", "辅修", "社团", "指导教师"]
        for kw in keywords:
            if kw in text: entities.append(kw)
        return list(set(entities))

    def initialize_data(self):
        # 尝试加载本地缓存
        if os.path.exists(VECTOR_DB_PATH):
            print("📂 检测到本地索引缓存，正在快速装载...")
            try:
                self.vector_retriever = FAISS.load_local(
                    VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True
                ).as_retriever(search_kwargs={"k": 6})
            except:
                pass

        print("📚 正在加载《学生手册》并构建多路召回图谱...")
        loader = PyPDFLoader(FILE_PATH)
        raw_pages = loader.load()
        
        cleaned_docs = []
        for page in raw_pages:
            txt = self.clean_text(page.page_content)
            if len(txt) > 10:
                cleaned_docs.append(Document(page_content=txt, metadata=page.metadata))
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150, separators=["\n\n", "第", "。", "！"]
        )
        self.documents = text_splitter.split_documents(cleaned_docs)
        
        if not self.vector_retriever:
            vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            vectorstore.save_local(VECTOR_DB_PATH)
            self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = 6
        except:
            pass

        for doc in self.documents:
            ents = self.extract_entities(doc.page_content)
            for ent in ents:
                self.entity_index[ent].append(doc)
        
        print(f"✅ 系统就绪! 片段:{len(self.documents)} 实体:{len(self.entity_index)}")

    # --- 单独封装重写逻辑，供前端调用 ---
    def rewrite_query(self, user_input):
        if not self.chat_history:
            return user_input # 没有历史，直接返回原问题，秒回！
        
        # ⚡️ 优化 Prompt：强制只输出结果，不要废话
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个查询重写工具。请结合历史对话，将用户的最新问题改写为一个完整的、独立的查询语句。\n"
                       "规则：\n"
                       "1. 补全缺失的主语或宾语（如将“它”替换为具体的社团名）。\n"
                       "2. 严禁输出“好的”、“改写如下”等废话。\n"
                       "3. 直接输出改写后的句子。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        context_chain = context_prompt | self.llm | StrOutputParser()
        return context_chain.invoke({
            "chat_history": self.chat_history,
            "question": user_input
        })

    def hybrid_retrieve(self, query: str, top_k=6) -> List[Document]:
        entities = self.extract_entities(query)
        graph_docs = []
        if entities:
            for ent in entities:
                if ent in self.entity_index:
                    for d in self.entity_index[ent]:
                        d.metadata['source'] = f'条款匹配({ent})'
                        graph_docs.append(d)
        
        vec_docs = self.vector_retriever.invoke(query)
        for d in vec_docs: d.metadata['source'] = '语义搜索'
        
        bm25_docs = []
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            for d in bm25_docs: d.metadata['source'] = '关键词'
        
        all_docs = graph_docs + vec_docs + bm25_docs
        unique_docs = []
        seen = set()
        for doc in all_docs:
            sig = doc.page_content[:50]
            if sig not in seen:
                unique_docs.append(doc)
                seen.add(sig)
        return unique_docs[:top_k]

# ================= 启动程序 =================
if __name__ == "__main__":
    bot = SHUCampusBot()
    bot.initialize_data()
    # 命令行测试逻辑...（略，主要看app.py）
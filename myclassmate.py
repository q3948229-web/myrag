import os
import time
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ================= 核心配置 (学校服务器) =================
API_KEY = "sk-kjcKGNQPBajYm3kHjpl7Kg"
BASE_URL = "http://10.10.22.76:4000/v1"

# 两个都用学校提供的模型
LLM_MODEL = "Qwen2.5-32B-Instruct"       # 用于对话
EMBED_MODEL = "Qwen3-Embedding-8B"       # 用于向量化

FILE_PATH = "2025年上海大学本科生学生手册.docx"
VECTOR_DB_PATH = "shu_handbook_server_index" # 本地缓存文件夹名
BATCH_SIZE = 10  # 🌟关键设置：每次只发 10 段给服务器，防止它 502
# =======================================================

def get_server_embeddings():
    """配置连接学校的 Embedding 模型"""
    return OpenAIEmbeddings(
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        model=EMBED_MODEL,
        check_embedding_ctx_length=False
    )

def init_vector_store():
    """智能初始化：有缓存读缓存，没缓存去连服务器"""
    embeddings = get_server_embeddings()

    # 1. 检查本地是否有缓存 
    if os.path.exists(VECTOR_DB_PATH):
        print("💾 检测到本地已存在向量库，正在直接加载...")
        print("   (本次启动不需要连接学校 Embedding 模型，速度极快)")
        try:
            return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"❌ 本地缓存损坏，准备重新构建: {e}")

    # 2. 如果没有缓存，开始重新构建
    print(f"📚 正在读取文档: {FILE_PATH} ...")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"找不到文件: {FILE_PATH}")

    loader = Docx2txtLoader(FILE_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    splits = text_splitter.split_documents(docs)
    total = len(splits)
    print(f"✂️  文档已切分为 {total} 个片段。")

    print(f"🚀 开始连接学校服务器 ({EMBED_MODEL}) 构建索引...")
    print(f"🛡️  启用安全模式：每批发送 {BATCH_SIZE} 段，防止服务器 502...")

    vectorstore = None
    
    # 🌟 核心优化：分批循环发送请求
    for i in range(0, total, BATCH_SIZE):
        batch = splits[i : i + BATCH_SIZE]
        print(f"   正在处理进度: {i}/{total} ...")
        
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            # 稍微休息一下，防止服务器判定攻击
            time.sleep(0.5) 
        except Exception as e:
            print(f"\n❌ 在第 {i} 段处发生错误: {e}")
            print("💡 建议：如果是 502，请联系管理员重启服务；如果是 429，请调小 BATCH_SIZE。")
            raise e

    # 3. 保存到本地
    print("💾 正在保存索引到本地硬盘...")
    vectorstore.save_local(VECTOR_DB_PATH)
    print("✅ 向量库构建完成并保存！下次运行将直接跳过此步。")
    return vectorstore

def main():
    try:
        # 1. 准备向量库 (Embedding 阶段)
        vectorstore = init_vector_store()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # 2. 准备对话模型 (Chat 阶段)
        print(f"🔌 正在连接对话模型: {LLM_MODEL} ...")
        llm = ChatOpenAI(
            api_key=API_KEY, base_url=BASE_URL, 
            model=LLM_MODEL, temperature=0.1
        )

        # 3. 构建 RAG 模板
        template = """你是一个专业的上海大学教务助手。请根据下方的【上下文】内容回答用户的问题。
        如果上下文中没有答案，请直接说“手册中未找到相关规定”，不要编造。
        
        【上下文】：
        {context}
        
        【问题】：
        {question}
        
        回答："""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\n🎉 系统就绪！完全连接学校服务器。(输入 q 退出)")
        
        # 4. 开始对话
        while True:
            query = input("\n🙋 提问: ")
            if query.lower() in ['q', 'exit']: break
            
            print("🤖 (正在思考)...")
            try:
                res = rag_chain.invoke(query)
                print(f"📖 回答:\n{res}")
            except Exception as e:
                print(f"❌ Chat请求失败: {e}")
                print("💡 可能是学校 Chat 模型挂了 (502)，请尝试重连。")

    except Exception as e:
        print(f"\n❌ 程序终止: {e}")

if __name__ == "__main__":
    main()
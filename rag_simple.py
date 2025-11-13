import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入必要的库
import faiss  # 用于高效的向量相似性搜索
import numpy as np  # 用于数值计算
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer  # 导入DPR模型和分词器
import requests  # 用于发送HTTP请求
import json  # 用于处理JSON数据
import os  # 用于文件和目录操作
import pickle  # 用于序列化和反序列化Python对象
import PyPDF2  # 用于PDF文件解析
import re  # 用于简单分词

# 加载预训练的DPR模型和分词器
# DPRContextEncoderTokenizer用于对文本进行分词处理，以便模型能够理解
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# DPRContextEncoder用于将文本转换为向量表示
context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# DPRQuestionEncoderTokeniz`er用于对问题进行分词处理
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# DPRQuestionEncoder用于将问题转换为向量表示
question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# 定义一个函数来解析PDF文件
def extract_text_from_pdf(pdf_path):
    """从PDF文件中提取文本"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"解析PDF文件时出错: {e}")
        return None
    return text

# 定义一个函数来将长文本切分为较小的块
def chunk_text(text, max_length=500):
    """将长文本切分为较小的块"""
    # 按句号分割文本
    sentences = text.split('。')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果加上当前句子超过最大长度，则保存当前块并开始新块
        if len(current_chunk + sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip() + "。")
            current_chunk = sentence
        else:
            current_chunk += sentence + "。"
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip() + "。")
    
    return chunks

# 定义一个函数来加载知识库文件
def load_knowledge_base(file_path):
    """加载知识库文件，支持txt和pdf格式"""
    knowledge_base = []
    
    # 检查文件扩展名
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.pdf':
        # 处理PDF文件
        print("正在解析PDF文件...")
        text = extract_text_from_pdf(file_path)
        if text is None:
            return []
        
        # 对PDF文本进行切片处理
        print("正在处理PDF文本...")
        chunks = chunk_text(text)
        knowledge_base.extend([chunk.strip() for chunk in chunks if chunk.strip()])
        
    elif ext.lower() == '.txt':
        # 处理TXT文件
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取所有行并存储在列表中
            lines = f.readlines()
        # 检查是否是按行分隔的知识库
        if len(lines) > 1 and all(len(line.strip()) < 500 for line in lines):
            # 多行短文本，每行作为一个知识条目
            knowledge_base = [line.strip() for line in lines if line.strip()]
        else:
            # 单个长文本文件，需要切片处理
            f.seek(0)  # 重置文件指针
            content = f.read().strip()
            chunks = chunk_text(content)
            knowledge_base = [chunk.strip() for chunk in chunks if chunk.strip()]
    else:
        print(f"不支持的文件格式: {ext}")
        return []
    
    return knowledge_base

# 定义一个函数来保存知识库向量
def save_knowledge_embeddings(embeddings, contexts, file_path):
    """保存知识库向量到文件"""
    data = {
        'embeddings': embeddings,
        'contexts': contexts
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"知识库向量已保存到 {file_path}")

# 定义一个函数来加载知识库向量
def load_knowledge_embeddings(file_path):
    """从文件加载知识库向量"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['contexts']

# 定义与TF-IDF相关的辅助函数
def tokenize_for_tfidf(text):
    """对文本进行简单分词，按英文串和单个中文字符拆分"""
    if not text:
        return []

    tokens = []
    buffer = []
    for ch in text.lower():
        if 'a' <= ch <= 'z' or ch.isdigit():
            buffer.append(ch)
        else:
            if buffer:
                tokens.append(''.join(buffer))
                buffer = []
            if '\u4e00' <= ch <= '\u9fff':
                tokens.append(ch)
    if buffer:
        tokens.append(''.join(buffer))
    return tokens


def build_tfidf_index(contexts):
    """手动构建TF-IDF索引，用于教学演示"""
    if not contexts:
        return None

    tokenized_docs = [tokenize_for_tfidf(text) for text in contexts]
    vocabulary = {}
    doc_freq = {}

    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    for token in doc_freq:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    total_docs = len(contexts)
    idf = {
        token: np.log((total_docs + 1) / (freq + 1)) + 1
        for token, freq in doc_freq.items()
    }

    tfidf_matrix = np.zeros((total_docs, len(vocabulary)), dtype=np.float32)

    for doc_idx, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue
        term_counts = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        max_count = max(term_counts.values()) if term_counts else 1

        for token, count in term_counts.items():
            token_idx = vocabulary[token]
            tf = count / max_count
            tfidf_matrix[doc_idx, token_idx] = tf * idf[token]

    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf_matrix = tfidf_matrix / norms

    return {
        'vocabulary': vocabulary,
        'idf': idf,
        'matrix': tfidf_matrix
    }


def tfidf_vectorize(text, tfidf_data):
    """将查询转换为TF-IDF向量"""
    tokens = tokenize_for_tfidf(text)
    if not tokens:
        return None

    vocabulary = tfidf_data['vocabulary']
    idf = tfidf_data['idf']
    vector = np.zeros(len(vocabulary), dtype=np.float32)

    term_counts = {}
    for token in tokens:
        if token in vocabulary:
            term_counts[token] = term_counts.get(token, 0) + 1

    if not term_counts:
        return None

    max_count = max(term_counts.values()) if term_counts else 1

    for token, count in term_counts.items():
        idx = vocabulary[token]
        tf = count / max_count
        vector[idx] = tf * idf[token]

    norm = np.linalg.norm(vector)
    if norm == 0:
        return None

    return vector / norm


def tfidf_retrieve_contexts(question, tfidf_data, contexts, k=1):
    """使用手动实现的TF-IDF余弦相似度检索"""
    if tfidf_data is None:
        return []

    matrix = tfidf_data['matrix']
    query_vector = tfidf_vectorize(question, tfidf_data)
    if query_vector is None:
        return []

    similarities = np.dot(matrix, query_vector)
    if similarities.size == 0:
        return []

    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [contexts[i] for i in top_k_indices]

# 定义一个函数来将文本转换为向量表示
def encode_contexts(contexts):
    # 使用分词器对文本进行分词处理，并返回张量格式的数据
    inputs = context_tokenizer(contexts, return_tensors='pt', padding=True, truncation=True)
    # 将分词后的数据输入到模型中，获取向量表示
    embeddings = context_model(**inputs).pooler_output.detach().numpy()
    # 返回向量表示
    return embeddings

# 定义一个函数来将问题转换为向量表示
def encode_question(question):
    """将问题文本转换为向量表示"""
    inputs = question_tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    question_embedding = question_model(**inputs).pooler_output.detach().numpy()
    return question_embedding

# 定义一个函数来使用Faiss高效检索最相关的知识
def faiss_retrieve_contexts(question_embedding, context_embeddings, contexts, k=1):
    """使用Faiss进行高效的向量相似性搜索"""
    # 使用Faiss进行相似性搜索
    # 创建一个索引，用于存储向量并进行相似性搜索
    index = faiss.IndexFlatIP(context_embeddings.shape[1])
    # 将知识库的向量添加到索引中
    index.add(context_embeddings)
    # 搜索与问题向量最相似的k个向量，并返回它们的索引
    _, indices = index.search(question_embedding, k)
    # 根据索引返回最相关的知识
    return [contexts[i] for i in indices[0]]

# 新增：定义一个使用简单余弦相似度进行检索的函数
def simple_retrieve_contexts(question_embedding, context_embeddings, contexts, k=1):
    """
    使用简单的余弦相似度进行检索（教学目的）。
    这个函数手动计算问题向量与所有知识向量的余弦相似度。
    """
    # 确保 question_embedding 是 2D 的，即使只有一个问题
    if question_embedding.ndim == 1:
        question_embedding = np.expand_dims(question_embedding, axis=0)

    # 计算问题向量的模长 (norm)
    question_norm = np.linalg.norm(question_embedding)

    # 计算知识库中所有上下文向量的模长
    context_norms = np.linalg.norm(context_embeddings, axis=1)

    # 计算问题向量与所有上下文向量的点积
    # .T 表示转置，使得矩阵乘法符合预期
    dot_products = np.dot(context_embeddings, question_embedding.T).flatten()

    # 计算余弦相似度： (A · B) / (||A|| * ||B||)
    # 添加一个极小值 epsilon 防止除以零
    epsilon = 1e-8
    similarities = dot_products / ((context_norms * question_norm) + epsilon)

    # 找到分数最高的 k 个向量的索引
    # np.argsort 会返回排序后的索引，我们取最后k个，然后反转顺序得到从大到小的索引
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    # 根据索引返回最相关的知识
    return [contexts[i] for i in top_k_indices]

# 定义一个函数来生成回答
def generate_answer(question, context):
    # 构造一个提示，告诉模型根据给定的上下文回答问题
    prompt = f"根据以下内容回答问题: {context}\n\n问题: {question}\n回答:"
    # 准备发送给Ollama API的数据
    data = {
        "model": "deepseek-r1:1.5b",  # 指定使用的模型
        "prompt": prompt,  # 提示内容
        "stream": False  # 不使用流式输出
    }
    # 发送POST请求到Ollama API
    response = requests.post("http://localhost:11434/api/generate", json=data)
    # 如果请求成功，返回生成的回答
    if response.status_code == 200:
        result = response.json()
        return result['response']
    # 如果请求失败，返回错误信息
    else:
        return "无法生成回答"

# 定义一个函数来初始化知识库
def initialize_knowledge_base():
    print("请选择知识库初始化方式:")
    print("1. 从文件读取数据并生成向量 (支持txt和pdf)")
    print("2. 从已有的向量化知识库加载")
    
    choice = input("请输入选项 (1 或 2): ").strip()
    
    if choice == "1":
        # 从文件读取数据
        knowledge_file = input("请输入知识库文件路径 (默认为 knowledge_base.txt): ").strip()
        if not knowledge_file:
            knowledge_file = "knowledge_base.txt"
        
        if not os.path.exists(knowledge_file):
            print(f"错误: 文件 {knowledge_file} 不存在")
            return None, None, None
            
        knowledge_base = load_knowledge_base(knowledge_file)
        if not knowledge_base:
            print("未能从文件中加载任何知识内容")
            return None, None, None
            
        print(f"已从 {knowledge_file} 加载 {len(knowledge_base)} 条知识")
        
        # 将知识库中的文本转换为向量表示
        print("正在生成向量表示...")
        context_embeddings = encode_contexts(knowledge_base)
        tfidf_data = build_tfidf_index(knowledge_base)
        if tfidf_data is None:
            print("构建TF-IDF索引失败")
            return None, None, None
        
        # 保存向量到文件
        save_file = input("请输入保存向量的文件名 (默认为 knowledge_embeddings.pkl): ").strip()
        if not save_file:
            save_file = "knowledge_embeddings.pkl"
        save_knowledge_embeddings(context_embeddings, knowledge_base, save_file)
        
        return context_embeddings, knowledge_base, tfidf_data
        
    elif choice == "2":
        # 从已有的向量化知识库加载
        embedding_file = input("请输入向量化知识库文件路径 (默认为 knowledge_embeddings.pkl): ").strip()
        if not embedding_file:
            embedding_file = "knowledge_embeddings.pkl"
            
        if not os.path.exists(embedding_file):
            print(f"错误: 文件 {embedding_file} 不存在")
            return None, None, None
            
        print("正在加载向量化知识库...")
        context_embeddings, knowledge_base = load_knowledge_embeddings(embedding_file)
        print(f"已从 {embedding_file} 加载 {len(knowledge_base)} 条知识")

        tfidf_data = build_tfidf_index(knowledge_base)
        if tfidf_data is None:
            print("构建TF-IDF索引失败")
            return None, None, None
        
        return context_embeddings, knowledge_base, tfidf_data
        
    else:
        print("无效选项")
        return None, None, None

# 定义多轮对话函数
def choose_interaction_mode():
    """让用户选择是执行完整的RAG还是仅做检索演示"""
    choice = ""
    while choice not in ["1", "2"]:
        print("\n请选择运行模式:")
        print("1. 检索 + 调用大模型生成回答 (完整RAG)")
        print("2. 仅检索并返回匹配内容 (不调用大模型)")
        choice = input("请输入选项 (1 或 2): ").strip()
    return "rag" if choice == "1" else "retrieve"


def chat_loop(context_embeddings, knowledge_base, tfidf_data, interaction_mode):
    print("\n开始多轮对话 (输入 '退出' 或 'quit' 结束对话):")
    
    # 让用户选择一次检索算法
    algo_choice = ""
    while algo_choice not in ["1", "2", "3"]:
        print("\n请选择检索算法:")
        print("1. Faiss (开源高效算法)")
        print("2. 简单余弦相似度 (教学算法)")
        print("3. 手写TF-IDF (教学算法)")
        algo_choice = input("请输入选项 (1、2 或 3): ").strip()

    if algo_choice == "1":
        print("您已选择 Faiss 算法。")
        retrieval_method = faiss_retrieve_contexts
        selected_algo = "faiss"
    else:
        if algo_choice == "2":
            print("您已选择简单的余弦相似度算法。")
            retrieval_method = simple_retrieve_contexts
            selected_algo = "simple"
        else:
            print("您已选择手写的TF-IDF算法。")
            retrieval_method = None
            selected_algo = "tfidf"

    while True:
        question = input("\n请输入你的问题: ").strip()
        
        if question.lower() in ['退出', 'quit', 'exit']:
            print("对话结束，再见！")
            break
            
        if not question:
            continue
            
        if selected_algo == "tfidf":
            retrieved_contexts = tfidf_retrieve_contexts(question, tfidf_data, knowledge_base, k=1)
        else:
            question_embedding = encode_question(question)
            retrieved_contexts = retrieval_method(question_embedding, context_embeddings, knowledge_base, k=1)
        
        if not retrieved_contexts:
            print("未能检索到相关知识，请尝试重新提问或更换算法。")
            continue

        if interaction_mode == "retrieve":
            print("检索到的内容:")
            for idx, context in enumerate(retrieved_contexts, start=1):
                print(f"[{idx}] {context}")
            continue

        answer = generate_answer(question, retrieved_contexts[0])
        
        # 打印问题、检索到的知识和生成的回答
        print(f"检索到的知识: {retrieved_contexts[0]}")
        print(f"生成的回答: {answer}")

# 主函数
def main():
    # 初始化知识库
    context_embeddings, knowledge_base, tfidf_data = initialize_knowledge_base()
    
    if context_embeddings is None or knowledge_base is None or tfidf_data is None:
        print("知识库初始化失败")
        return

    interaction_mode = choose_interaction_mode()
    
    # 开始多轮对话
    chat_loop(context_embeddings, knowledge_base, tfidf_data, interaction_mode)

# 程序入口点
if __name__ == "__main__":
    # 调用主函数
    main()
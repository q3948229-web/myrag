# 上海大学 2025 本科生手册 智能问答系统 (SHU Handbook RAG)

## 🌟 项目简介
本项目是一个针对 **2025年上海大学本科生学生手册** 开发的高级检索增强生成 (RAG) 系统。系统旨在解决传统文本检索在面对规章制度时“语义模糊”和“逻辑断裂”的问题，通过 **Hybrid RAG (混合检索)** 技术，为用户提供像查字典一样精准、像聊天一样自然的问答体验。

---

## 🚀 核心技术架构

### 1. 混合检索引擎 (Hybrid Retrieval)
系统摒弃了单一的向量检索，采用了“三路并发”的召回策略：
- **精确 SQL 匹配**：针对“第 X 条”、“第 X 章”等强逻辑查询，直接从 SQLite 数据库提取官方原始条目。
- **关键词/BM25 匹配**：针对“转专业”、“绩点”等高频核心业务词汇，通过实体索引和统计学模型确保召回。
- **语义向量检索**：利用 `FAISS` 向量库，处理用户模糊描述的意图（如“想休学怎么办”）。

### 2. 深度上下文理解
- **查询重写 (Query Rewriting)**：结合多轮对话历史，自动补全用户问题中的指代消解（如将“怎么申请”优化为“怎么申请转专业”）。
- **语义去重与重排**：对多路召回的资料进行智能去重（重复率检测）和得分过滤。

---

## 📂 项目结构说明

- [rag_hybrid.py](rag_hybrid.py): **系统主入口**。采用原生的 OpenAI SDK + SQLite + FAISS 实现，执行效率极高。
- [test12.py](test12.py): **LangChain 实验版**。引入了 `BM25` 和 `BaseRetriever` 框架，支持更复杂的文档解析逻辑。
- [config_api.py](config_api.py): 集中管理 LLM (如 Qwen2.5) 和 Embedding 型号及 API 地址。
- **data/processed/**:
    - `handbook.db`: 结构化手册数据库。
    - `vector_index.bin`: 高性能 FAISS 向量索引。
    - `metadata.pkl`: 存储向量对应的原始文本分块。

---

## 🛠️ 环境搭建与启动

### 1. 环境准备
推荐使用 Conda 管理环境：
```powershell
# 创建并激活环境
conda create -n myrag_env python=3.11 -y
conda activate myrag_env

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API (config_api.py)
```python
API_KEY = "你的密钥"
BASE_URL = "API请求地址"
MODEL_LLM = "Qwen2.5-32B-Instruct"
MODEL_EMBEDDING = "Qwen3-Embedding"
```

### 3. 运行问答机器人
```powershell
python rag_hybrid.py
```

---

## 📝 数据处理流水线
本项目的优势在于极其精细的数据预处理逻辑：
1. **结构化提取**：将手册解析为“章-节-条”的层级结构并存入 SQL。
2. **文本分块修复**：通过 LLM 语义识别修复 PDF 解析过程中产生的断句问题。
3. **多维索引构建**：同步构建关系型索引 (B-Tree/Hash) 与向量索引。

---

## ⚠️ 注意事项
- 问答结果取决于参考资料的完整性，如遇“资料不足”，系统会如实告知。
- 建议使用 Python 3.11+ 以获得最佳的库兼容性（特别是 `networkx` 和数据处理工具）。



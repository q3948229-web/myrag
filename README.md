# Shanghai University Undergraduate Handbook RAG (2025)

## 项目简介
本项目是一个针对 **2025年上海大学本科生学生手册** 开发的混合检索增强生成 (Hybrid RAG) 系统。它结合了向量检索与结构化数据处理，旨在为学生和教职工提供准确的手册内容查询服务。

## 核心功能
- **混合检索**: 结合语义向量搜索与关键词匹配。
- **结构化数据映射**: 将学生手册内容转化为 SQL 结构，提高规章制度查询的精准度。
- **完整流水线**: 包含数据清洗、分块修复、语义细化及索引构建。

## 项目结构
- `rag_hybrid.py`: 系统主入口，执行混合检索与问答逻辑。
- `config_api.py`: 配置文件，包含 LLM/Embedding API 密钥及地址。
- `data/`:
    - `raw/`: 原始手册文件 (MD, PDF, DOCX)。
    - `processed/`: 经过清洗、修复和向量化的中间数据。
- `scripts/`: 数据处理工具集，包括清洗、修复、重建等脚本。

## 快速开始
1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```
2. **配置 API**:
   在 `config_api.py` 中填入你的 `API_KEY` 和 `BASE_URL`。
3. **运行系统**:
   ```bash
   python rag_hybrid.py
   ```

## 数据处理流程
1. **清洗数据**: 使用 `scripts/clean_handbook.py` 处理原始文本。
2. **知识细化**: 通过 `scripts/refine_data.py` 和 `scripts/recover_chunks.py` 修复损坏的分块。
3. **重建基础**: 如需从 docx 重新构建系统，运行 `scripts/rebuild_data_from_docx.py`。

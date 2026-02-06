import os
import json
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from openai import OpenAI

# 导入自定义机器人
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_hybrid import SHUHandbookBot
from config_api import API_KEY, BASE_URL, MODEL_LLM

# ================= 配置区 =================
TEST_SET_PATH = "d:/myrag/eval_test_set.json"
RAW_RESULTS_PATH = "d:/myrag/eval_results_raw.json"
FINAL_REPORT_PATH = "d:/myrag/eval_final_report.csv"

# RPM 限制配置 (RPM 10)
REQ_INTERVAL = 7.0  # 每次请求间隔 7 秒，保守起见

class RateLimitedEvaluator:
    def __init__(self):
        self.bot = SHUHandbookBot()
        
    def collect_data(self):
        """第一阶段：推理与采集数据"""
        if not os.path.exists(TEST_SET_PATH):
            print(f"错误: 找不到测试集 {TEST_SET_PATH}")
            return None
            
        with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
            
        # 如果已经有部分结果，支持断点续传（可选手动清理）
        results = []
        if os.path.exists(RAW_RESULTS_PATH):
            with open(RAW_RESULTS_PATH, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        start_idx = len(results)
        print(f"开始推理采集... 已完成: {start_idx}/{len(test_set)}")
        
        for i in range(start_idx, len(test_set)):
            item = test_set[i]
            question = item['question']
            print(f"[{i+1}/{len(test_set)}] 正在处理问题: {question}")
            
            # 执行 RAG 流程
            # 注意: bot.ask 内部会发起: 1. Rewrite LLM, 2. Embedding, 3. Answer LLM
            # 这可能在一次问答中消耗 3 次请求
            ans, context, rewritten = self.bot.ask(question)
            
            results.append({
                "question": question,
                "answer": ans,
                "contexts": [context], # Ragas 需要 list of strings
                "ground_truth": item['ground_truth']
            })
            
            # 保存进度
            with open(RAW_RESULTS_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 为了严防 RPM 10，由于 ask() 内部有多步调用，这里间隔要久一点
            print(f"等待 {REQ_INTERVAL * 2}s 以维护 RPM 限制...")
            time.sleep(REQ_INTERVAL * 2) 
            
        return results

def run_ragas_evaluation(data_list):
    """第二阶段：使用 Ragas 进行评分"""
    print("\n--- 开始 Ragas 评分阶段 ---")
    
    # 将数据转换为 Dataset 格式
    df = pd.DataFrame(data_list)
    dataset = Dataset.from_pandas(df)
    
    # 由于 Ragas 内部调用 LLM 且不支持直接在内部 sleep
    # 我们通过配置自定义的 OpenAI 客户端（LangChain 兼容）来规避，
    # 但简单起见，如果数据量不大，我们可以利用 Ragas 的 evaluate 参数。
    
    # 这里的难点是 Ragas 默认会并发请求。
    # 我们需要通过环境变量或配置限制其并发数。
    
    try:
        # 使用你现有的配置
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.run_config import RunConfig
        
        # 为 Ragas 配置带限速的 LLM
        eval_llm = ChatOpenAI(
            model=MODEL_LLM,
            api_key=API_KEY,      # 显式传递参数名 api_key
            base_url=BASE_URL,
            max_retries=10,       # 增加重试次数
            timeout=180,          # 增加超时限制
        )
        
        # Ragas 评分也需要 Embedding 模型
        from config_api import MODEL_EMBEDDING
        eval_embeddings = OpenAIEmbeddings(
            model=MODEL_EMBEDDING,
            api_key=API_KEY,
            base_url=BASE_URL,
            chunk_size=1
        )
        
        # 核心修改：使用 RunConfig 限制并发数为 1 (替代已废弃的 is_async)
        # 这将确保遵循 RPM 10 限制
        config = RunConfig(max_workers=1, timeout=180)
        
        # 执行评估
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=config  # 使用配置对象控制执行行为
        )
        
        print("\n✅ 评估完成！总结指标：")
        print(result)
        
        # 保存详细报表
        result_df = result.to_pandas()
        result_df.to_csv(FINAL_REPORT_PATH, index=False, encoding='utf-8-sig')
        print(f"结果已保存至: {FINAL_REPORT_PATH}")
        
    except Exception as e:
        print(f"评分阶段出错: {e}")
        print("提示: 可能是由于 Ragas 并发请求触发了 RPM 限制。")

if __name__ == "__main__":
    evaluator = RateLimitedEvaluator()
    
    # 步骤 1: 采集 RAG 输出
    raw_data = evaluator.collect_data()
    
    # 步骤 2: 运行评分
    if raw_data:
        run_ragas_evaluation(raw_data)

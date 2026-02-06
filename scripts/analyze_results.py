import pandas as pd
import numpy as np
import os

def analyze_eval_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # 读取结果
    df = pd.read_csv(file_path)
    
    # 定义需要统计的指标列
    metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    
    # 过滤掉非数值列（以防万一）
    available_metrics = [m for m in metrics if m in df.columns]
    
    # 计算基本统计指标
    stats = df[available_metrics].describe().T
    stats['median'] = df[available_metrics].median()
    
    # 计算达标率 (>= 0.8)
    for m in available_metrics:
        stats.loc[m, 'pass_rate(>=0.8)'] = (df[m] >= 0.8).mean()

    # 打印报表
    print("\n" + "="*50)
    print("      SHU Handbook RAG Evaluation Summary")
    print("="*50)
    
    output_df = stats[['mean', 'std', 'median', 'min', 'max', 'pass_rate(>=0.8)']]
    print(output_df.to_string(float_format=lambda x: "{:.4f}".format(x)))
    
    print("\n" + "="*50)
    print("      Correlation Matrix")
    print("="*50)
    print(df[available_metrics].corr().to_string(float_format=lambda x: "{:.4f}".format(x)))

    # 找出表现最差的 Top 3 问题 (按均值总分排序)
    df['total_score'] = df[available_metrics].mean(axis=1)
    bad_cases = df.sort_values(by='total_score').head(3)
    
    print("\n" + "="*50)
    print("      Top 3 Weakest Cases (Lowest Average)")
    print("="*50)
    for i, row in bad_cases.iterrows():
        print(f"Q: {row['user_input']}")
        print(f"Scores: F:{row.get('faithfulness',0):.2f}, AR:{row.get('answer_relevancy',0):.2f}, CP:{row.get('context_precision',0):.2f}, CR:{row.get('context_recall',0):.2f}")
        print("-" * 20)

if __name__ == "__main__":
    analyze_eval_results(r"d:\myrag\eval_final_report.csv")

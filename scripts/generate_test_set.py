import os
import re
import json
import time
from docx import Document
from openai import OpenAI

# 导入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_api import API_KEY, BASE_URL, MODEL_LLM
except ImportError:
    API_KEY = "your-api-key-here"
    BASE_URL = "http://localhost:4000/v1"
    MODEL_LLM = "Qwen2.5-32B-Instruct"

# 配置路径
DOCX_PATH = "d:/myrag/data/raw/2025年上海大学本科生学生手册.docx"
OUTPUT_PATH = "d:/myrag/eval_test_set.json"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def parse_docx_to_articles(path):
    """参考 rebuild_data_from_docx.py 的解析逻辑，提取核心条款"""
    print(f"正在读取文档: {path} ...")
    doc = Document(path)
    
    current_doc = ""
    current_chapter = ""
    current_article_title = ""
    current_article_content = []
    articles = []
    
    def save_article():
        if current_article_content and current_doc:
            full_content = "".join(current_article_content).strip()
            if len(full_content) > 50:  # 过滤掉太短的无意义段落
                articles.append({
                    "context": f"【{current_doc} - {current_chapter}】\n{current_article_title}\n{full_content}",
                    "raw_text": full_content
                })
            current_article_content.clear()

    re_chapter = re.compile(r'^(第[一二三四五六七八九十百]+章)\s*(.*)')
    re_article = re.compile(r'^(第[一二三四五六七八九十百]+条)\s*(.*)')

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text: continue
            
        if para.style.name == 'Heading 1':
            save_article()
            current_doc = text
            continue

        if re_chapter.match(text) or para.style.name == 'Heading 2':
            save_article()
            current_chapter = text
            continue

        if re_article.match(text):
            save_article()
            current_article_title = text
            continue
        
        if current_doc:
            current_article_content.append(text)
            
    save_article()
    print(f"解析完成，提取到 {len(articles)} 条有效内容。")
    return articles

def generate_cases_from_article(article_data):
    """调用 LLM 为单个条款生成测试用例"""
    prompt = f"""你是一个教育专家。请根据以下《上海大学学生手册》的条款内容，编写一个真实的学生咨询问题及标准答案。

内容：
{article_data['context']}

要求：
1. 问题(question)：模仿学生真实的口吻，不要直接照抄条文。
2. 标准答案(ground_truth)：必须源自上述内容，表述准确。
3. 返回格式：严格使用 JSON 格式，如下：
{{"question": "问题内容", "ground_truth": "答案内容"}}

请直接输出 JSON，不要有任何解释。"""

    try:
        response = client.chat.completions.create(
            model=MODEL_LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        # 简单清洗，防止模型输出包含 ```json
        content = re.sub(r'```json\s*|\s*```', '', content)
        return json.loads(content)
    except Exception as e:
        print(f"生成失败: {e}")
        return None

def main():
    # 1. 解析文档
    all_articles = parse_docx_to_articles(DOCX_PATH)
    
    # 2. 抽样生成（为了节省 Token 和绕过限制，我们挑选 20 条，或者每隔固定条数取一条）
    # 这里我们采用等间距采样，取 30 条
    sample_size = 30
    step = max(1, len(all_articles) // sample_size)
    sampled_articles = all_articles[::step][:sample_size]
    
    test_set = []
    print(f"开始生成测试集 (限制 RPM: 10).预计耗时: {len(sampled_articles) * 7} 秒")
    
    for i, article in enumerate(sampled_articles):
        print(f"[{i+1}/{len(sampled_articles)}] 正在为条款生成问题...")
        
        case = generate_cases_from_article(article)
        if case:
            test_set.append(case)
            # 自动保存一次，防止中断
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(test_set, f, ensure_ascii=False, indent=2)
        
        # 严格遵守 RPM 10 限制：每分钟最多10次，即每次间隔 6.5 秒
        if i < len(sampled_articles) - 1:
            time.sleep(6.5)

    print(f"✅ 生成完成！共生成 {len(test_set)} 条测试用例。")
    print(f"文件保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

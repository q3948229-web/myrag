import socket
import json
import time
import os
import sys
from openai import OpenAI

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_api import API_KEY, BASE_URL, MODEL_LLM, MODEL_EMBEDDING
    # 提取主机和端口
    from urllib.parse import urlparse
    parsed = urlparse(BASE_URL)
    _host = parsed.hostname
    _port = parsed.port or (80 if parsed.scheme == "http" else 443)
except ImportError:
    API_KEY = "sk-..."
    BASE_URL = "http://..."
    MODEL_LLM = "Qwen2.5-32B-Instruct"
    MODEL_EMBEDDING = "Qwen3-Embedding-8B"
    _host = "127.0.0.1"
    _port = 4000

# ================= 配置区 =================
CONFIG = {
    "api_key": API_KEY,
    "base_url": BASE_URL,
    "host": _host,
    "port": _port
}

client = OpenAI(
    api_key=CONFIG["api_key"],
    base_url=CONFIG["base_url"]
)

def test_tcp_connectivity():
    print(f"\n[测试 1/3] 正在测试 TCP 连通性 ({CONFIG['host']}:{CONFIG['port']})...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        start_time = time.time()
        s.connect((CONFIG['host'], CONFIG['port']))
        elapsed = (time.time() - start_time) * 1000
        print(f"✅ TCP 端口开放，连接耗时: {elapsed:.2f}ms")
        return True
    except Exception as e:
        print(f"❌ TCP 连接失败: {e}")
        return False
    finally:
        s.close()

def test_auth_and_chat():
    print(f"\n[测试 2/3] 正在测试身份验证与 Chat 接口 (模型: {MODEL_LLM})...")
    try:
        response = client.chat.completions.with_raw_response.create(
            model=MODEL_LLM,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5
        )
        print("✅ 身份验证通过，服务器返回原始数据:")
        print(json.dumps(response.parse().model_dump(), indent=2, ensure_ascii=False))
    except Exception as e:
        print("❌ Chat 接口/身份验证失败！原始错误信息如下:")
        # 如果是 OpenAI 错误对象，打印其 body
        if hasattr(e, 'body'):
            print(json.dumps(e.body, indent=2, ensure_ascii=False))
        else:
            print(str(e))

def test_embedding_raw():
    print(f"\n[测试 3/3] 正在测试 Embedding 接口原始返回 (模型: {MODEL_EMBEDDING})...")
    try:
        response = client.embeddings.with_raw_response.create(
            input=["test"],
            model=MODEL_EMBEDDING
        )
        full_res = response.parse().model_dump()
        # 截断向量数据以方便观察其他元数据
        if 'data' in full_res and len(full_res['data']) > 0:
            full_res['data'][0]['embedding'] = f"[Vector with length {len(full_res['data'][0]['embedding'])} ...]"
        
        print("✅ Embedding 响应成功，服务器原始元数据:")
        print(json.dumps(full_res, indent=2, ensure_ascii=False))
    except Exception as e:
        print("❌ Embedding 接口测试失败:")
        if hasattr(e, 'body'):
            print(json.dumps(e.body, indent=2, ensure_ascii=False))
        else:
            print(str(e))

if __name__ == "__main__":
    print("="*50)
    print("API 深度诊断测试 (含 TCP、认证、原始响应)")
    print("="*50)
    
    if test_tcp_connectivity():
        test_auth_and_chat()
        test_embedding_raw()
    
    print("\n诊断完成。")

# fastllm
基于 xiaoapi 提供大语言模型调用接口


## 安装
```
pip install -r requirements.txt
```

## 配置
在 application/settings 中 添加以下配置：
```python
EMBEDDINGS_ENABLED = True
EMBEDDINGS_MODEL_NAME = "bge-large-zh-v1.5"
EMBEDDINGS_MODEL_PATH = r"E:\WorkSpace\LLMWorkSpace\Models\Embedding\bge-large-zh-v1.5"
RERANKER_ENABLED = True
RERANKER_MODEL_NAME = "bge-reranker-base"
RERANKER_MODEL_PATH = r"E:\WorkSpace\LLMWorkSpace\Models\reranker\bge-reranker-base"
```

## 启动
```
python manage.py run-server
```
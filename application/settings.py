"""
FastAPI settings for project.
"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 注意：不要在生产中打开调试运行!
DEBUG = False

####################
# PROJECT SETTINGS #
####################
TITLE = "Fast LLM"
DESCRIPTION = "基于 xiaoapi 提供大语言模型调用接口"
VERSION = "0.0.1"


############
# UVICORN #
############
# 监听主机IP，默认开放给本网络所有主机
HOST = "0.0.0.0"
# 监听端口
PORT = 9000
# 工作进程数
WORKERS = 1


#######
# LLM #
#######
EMBEDDINGS_ENABLED = True
EMBEDDINGS_MODEL_NAME = "bge-large-zh-v1.5"
EMBEDDINGS_MODEL_PATH = r"E:\WorkSpace\LLMWorkSpace\Models\Embedding\bge-large-zh-v1.5"
RERANKER_ENABLED = True
RERANKER_MODEL_NAME = "bge-reranker-base"
RERANKER_MODEL_PATH = r"E:\WorkSpace\LLMWorkSpace\Models\reranker\bge-reranker-base"


##############
# MIDDLEWARE #
##############
# List of middleware to use. Order is important; in the request phase, these
# middleware will be applied in the order given, and in the response
# phase the middleware will be applied in reverse order.
MIDDLEWARES = [
    "xiaoapi.middleware.register_request_log_middleware",
]


############
# LIFESPAN #
############
EVENTS = [
    "modules.fastllm.event.create_embeddings_worker" if EMBEDDINGS_ENABLED else None,
    "modules.fastllm.event.create_reranker_worker" if RERANKER_ENABLED else None,
    "modules.fastllm.event.create_torch_gc_task",
]

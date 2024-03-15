from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from application.settings import EMBEDDINGS_MODEL_PATH, EMBEDDINGS_MODEL_NAME, RERANKER_MODEL_PATH, \
    RERANKER_MODEL_NAME
from modules.fastllm.model_loader.embeddings.embeddings_worker import EmbeddingsWorker
from modules.fastllm.model_loader.reranker.reranker_worker import RerankerWorker
from modules.fastllm.model_loader.utils import detect_device, torch_gc


async def create_embeddings_worker(app: FastAPI, status: bool):
    """
    创建embeddings worker，并加载模型
    """
    if status:
        worker = EmbeddingsWorker(
            model_path=EMBEDDINGS_MODEL_PATH,
            model_name=EMBEDDINGS_MODEL_NAME,
            device=detect_device()
        )

        app.state.embeddings = worker
    else:
        pass


async def create_reranker_worker(app: FastAPI, status: bool):
    """
    创建reranker worker
    首次调用时加载模型
    """
    if status:
        worker = RerankerWorker(
            model_path=RERANKER_MODEL_PATH,
            model_name=RERANKER_MODEL_NAME,
            device=detect_device()
        )

        app.state.reranker = worker
    else:
        pass


async def create_torch_gc_task(app: FastAPI, status: bool):
    """
    创建清理缓存的定时任务，每隔1小时执行一次
    """
    if status:
        scheduler = BackgroundScheduler()
        scheduler.add_job(torch_gc, 'interval', minutes=60)
        scheduler.start()

        app.state.scheduler = scheduler
    else:
        app.state.scheduler.shutdown()

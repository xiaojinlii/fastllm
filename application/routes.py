from fastapi import FastAPI

from modules.fastllm.routes import router as fastllm_router


def register_routes(app: FastAPI):
    """
    注册路由
    """

    app.include_router(fastllm_router, tags=["大语言模型"])

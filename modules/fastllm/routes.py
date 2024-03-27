from typing import List

from fastapi import APIRouter, Depends, Request, Body
from starlette.responses import JSONResponse

from xiaoapi.core import CustomException
from application.settings import EMBEDDINGS_ENABLED, RERANKER_ENABLED
from modules.fastllm.model_loader.embeddings.embeddings_worker import EmbeddingsWorker
from modules.fastllm.model_loader.reranker.reranker_worker import RerankerWorker

router = APIRouter()


def embeddings_getter(request: Request) -> EmbeddingsWorker:
    if not EMBEDDINGS_ENABLED:
        raise CustomException(
            msg="请先开启 embeddings 模型的创建！",
            desc="请启用 application/settings.py: EMBEDDINGS_ENABLED"
        )
    return request.app.state.embeddings


def reranker_getter(request: Request) -> RerankerWorker:
    if not RERANKER_ENABLED:
        raise CustomException(
            msg="请先开启 reranker 模型的创建！",
            desc="请启用 application/settings.py: RERANKER_ENABLED"
        )
    return request.app.state.reranker


@router.post("/worker_embed_documents", summary="将字符串列表转为向量列表")
async def worker_embed_documents(
        texts: List[str] = Body(description="需要embedding的字符串列表", examples=[["嗨", "哈喽"]]),
        worker: EmbeddingsWorker = Depends(embeddings_getter)
):
    embeddings = worker.embed_documents(texts)
    return JSONResponse(content=embeddings)


@router.post("/worker_embed_query", summary="将查询语句转为向量")
async def worker_embed_query(
        text: str = Body(description="需要embedding的字符串", examples=["嗨"]),
        worker: EmbeddingsWorker = Depends(embeddings_getter)
):
    embeddings = worker.embed_query(text)
    return JSONResponse(content=embeddings)


@router.post("/worker_get_embeddings", summary="将字符串列表转为向量列表，并进行编码")
async def worker_get_embeddings(
        texts: List[str] = Body(description="需要embedding的字符串列表", examples=[["嗨", "哈喽"]]),
        encoding_format: str = Body(description="编码格式，仅支持base64，其他格式不进行编码直接返回", examples=["base64"]),
        worker: EmbeddingsWorker = Depends(embeddings_getter)
):
    embeddings = worker.get_embeddings(texts, encoding_format)
    return JSONResponse(content=embeddings)


@router.post("/worker_compute_score", summary="计算评分")
async def worker_compute_score(
        sentence_pairs: List[List[str]] = Body(description="需要打分的字符串列表对", examples=[[["哈喽", "嗨"], ["你在做什么", "你干啥呢"]]]),
        worker: RerankerWorker = Depends(reranker_getter)
):
    scores = worker.compute_score(sentence_pairs)
    return JSONResponse(content=scores)


@router.post("/worker_compute_score_by_query", summary="根据查询语句计算评分")
async def worker_compute_score_by_query(
        query: str = Body(description="源字符串", examples=["嗨"]),
        texts: List[str] = Body(description="目标字符串列表，根据query进行一一打分", examples=[["嗨", "哈喽"]]),
        worker: RerankerWorker = Depends(reranker_getter)
):
    embeddings = worker.compute_score_by_query(query, texts)
    return JSONResponse(content=embeddings)

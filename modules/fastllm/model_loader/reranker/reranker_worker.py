from typing import List, Literal

from xiaoapi.core import logger


class RerankerWorker:
    def __init__(
            self,
            model_path: str,
            model_name: str,
            device: str = Literal["mps", "cuda", "cpu"],
            batch_size: int = 32,
            num_workers: int = 0,
    ):
        logger.info(f"Loading the reranker model {model_name} ...")

        self.batch_size = batch_size
        self.num_workers = num_workers

        try:
            import sentence_transformers

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self.model = sentence_transformers.CrossEncoder(model_name=model_path, max_length=1024, device=device)

    def compute_score(self, sentence_pairs: List[List[str]]) -> List[float]:
        scores = self.model.predict(
            sentences=sentence_pairs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            convert_to_tensor=True
        )
        scores = scores.cpu()
        results = scores.numpy().tolist()
        return results

    def compute_score_by_query(self, query: str, texts: List[str]) -> List[float]:
        sentence_pairs = [[query, text] for text in texts]
        return self.compute_score(sentence_pairs)


if __name__ == "__main__":
    from modules.fastllm.model_loader.utils import detect_device

    worker = RerankerWorker(
        model_path=r"E:\WorkSpace\LLMWorkSpace\Models\reranker\bge-reranker-base",
        model_name="bge-reranker-base",
        device=detect_device()
    )

    result = worker.compute_score([["哈喽", "嗨"], ["你在做什么", "你干啥呢"]])
    print(result)
    result = worker.compute_score_by_query("哈喽", ["嗨", "hello", "hi"])
    print(result)

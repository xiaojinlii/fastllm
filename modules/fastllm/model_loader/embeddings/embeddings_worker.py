from typing import List, Literal

from xiaoapi.core import logger


def encode_base64(embeddings: List[List[float]]) -> List[str]:
    import base64
    import numpy as np

    encoded_data_list = []
    for embedding in embeddings:
        # 将嵌套浮点数列表转换为numpy数组，dtype设为float32
        array = np.array(embedding, dtype=np.float32)
        # 将numpy数组转化为字节串
        data_bytes = array.tobytes()
        # 使用base64进行编码
        encoded_data = base64.b64encode(data_bytes)
        # 将编码后的字节串转为str并添加到结果列表中
        encoded_data_list.append(encoded_data.decode('utf-8'))
    return encoded_data_list


class EmbeddingsWorker:
    def __init__(
            self,
            model_path: str,
            model_name: str,
            device: str = Literal["mps", "cuda", "cpu"],
    ):
        logger.info(f"Loading the embeddings model {model_name} ...")

        if 'bge-' in model_name:
            from .huggingface import HuggingFaceBgeEmbeddings
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_path,
                model_kwargs={'device': device}
            )
            if model_name == "bge-large-zh-noinstruct":
                self.embeddings.query_instruction = ""
        else:
            from .huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': device}
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def get_embeddings(self, texts: List[str], encoding_format: str = None) -> List[List[float]]:
        normalized_embeddings = self.embed_documents(texts)
        if encoding_format == "base64":
            out_embeddings = encode_base64(normalized_embeddings)
        else:
            out_embeddings = normalized_embeddings
        return out_embeddings


if __name__ == "__main__":
    from modules.fastllm.model_loader.utils import detect_device

    worker = EmbeddingsWorker(
        model_path=r"E:\WorkSpace\LLMWorkSpace\Models\Embedding\bge-large-zh-v1.5",
        model_name="bge-large-zh-v1.5",
        device=detect_device()
    )

    result = worker.embed_query("哈喽")
    print(result)

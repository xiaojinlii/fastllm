import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def huggingface(model_path, pairs):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)


def sentence(model_path, pairs):
    model = CrossEncoder(model_name=model_path, max_length=1024, device="cpu")

    results = model.predict(sentences=pairs,
                            # batch_size=self.batch_size,
                            #  show_progress_bar=self.show_progress_bar,
                            # num_workers=self.num_workers,
                            #  activation_fct=self.activation_fct,
                            #  apply_softmax=self.apply_softmax,
                            convert_to_tensor=True
                            )

    print(results)


if __name__ == '__main__':
    model_path = r"E:\WorkSpace\LLMWorkSpace\Models\reranker\bge-reranker-base"

    pairs = [
        ['哈喽', '嗨'],
        ['早上好', '晚上好'],
        ['你在做什么', '你干啥呢'],
        ['你好', '你好'],
    ]

    # 这两个接口结果不同，huggingface的结果值偏大，sentence的结果值在0~1范围内
    huggingface(model_path, pairs)
    sentence(model_path, pairs)

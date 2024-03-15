import requests

sentences = [
    ['哈喽', '嗨'],
    ['早上好', '晚上好'],
    ['你在做什么', '你干啥呢'],
    ['你好', '你好'],
]
response = requests.post("http://127.0.0.1:21021/worker_compute_score", json=sentences)
print(response.json())


data = {
    "query": "哈喽",
    "texts": ["哈喽", "嗨", "你好", "早上好"]
}
response = requests.post("http://127.0.0.1:21021/worker_compute_score_by_query", json=data)
print(response.json())

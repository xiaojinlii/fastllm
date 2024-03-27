import requests

texts = [
    "哈喽",
    "嗨"
]
response = requests.post("http://127.0.0.1:21021/worker_embed_documents", json=texts)
print(response.json())

text = "哈喽"
response = requests.post("http://localhost:21021/worker_embed_query", json=text)
print(response.json())

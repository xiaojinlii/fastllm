## linux命令
- 每隔0.1秒刷新一次显存使用情况
    ```shell
    watch -n 0.1 -d nvidia-smi
    ```
- 0.5s刷新一次
	```shell
    top -d 0.5
	```
- 实时查看某个进程的情况
	```shell
    top -p pid
	```
- 查看某个进程的内存占用
	```shell
    ps -p pid -o rss= | awk '{ printf "%.2f MB\n", $1 / 1024 }'
	```




## 内存、显存占用
### 单独启动fastapi，不加载任何模型
1个worker	内存占用：402MB
2个worker	内存占用：parent进程399MB，每个worker进程402MB
3个worker	内存占用：parent进程399MB，每个worker进程402MB
n个worker	内存占用：parent进程399MB，每个worker进程402MB

### 只有bge-large-zh-v1.5模型
1个worker：
    加载模型后显存占用：1932MB
    加载模型后内存占用：3138MB，3127MB，3124MB

    上传54个知识库文档后：
        加载模型后内存占用：3127MB
        gc前：
            显存占用：2786MB
            内存占用：3325MB
        gc后：
            显存占用：2312MB
            内存占用：3325MB
    
    jmemer->worker_embed_documents 1次：
        加载模型后内存占用：3124MB
        gc前：
            显存占用：2314MB
            内存占用：3307MB
        gc后：
            显存占用：2312MB
            内存占用：3307MB

2个worker：
    加载模型后每个worker进程显存占用：1932MB
    加载模型后内存占用：parent进程400MB，每个worker进程大约3125MB

    jmemer->worker_embed_documents 20次：
        加载模型后内存占用：3125MB
        gc前：
            显存占用：2314MB（每个worker）
            内存占用：3308MB（每个worker）
        gc后：
            显存占用：2312MB（每个worker）
            内存占用：3308MB（每个worker）


### 只有bge-reranker-base模型
1个worker：
	启动后不占用显存
	启动后内存占用：1884MB

	首次会加载模型较慢，A4000大约5s，第2次0.9s
	jmemer->worker_compute_score_by_query 1次：
	    gc前：
	        显存占用：2156MB
	        内存占用：3206MB
	    gc后：
	        显存占用：2154MB
	        内存占用：3206MB


​	        
## 并发
测试方式：1s内发送完n个请求
标准：90%请求在3s内响应
### RTX 4090
#### 只加载bge-large-zh-v1.5模型
1workers
	worker_embed_query：130，性能瓶颈在cpu，gpu占16%
	worker_embed_documents：50，性能瓶颈在cpu，gpu占61%，gpu占用跟字数成正比
2workers
	worker_embed_query：250，性能瓶颈在cpu，gpu占61%
	worker_embed_documents：55，性能瓶颈在gpu，gpu占100%
3workers
	worker_embed_query：370，性能瓶颈在cpu，gpu占90%
	worker_embed_documents：50，性能瓶颈在gpu，gpu占100%
4workers
	worker_embed_query：500，gpu占99%
#### 只加载bge-reranker-base模型
1workers
	worker_compute_score_by_query：220，性能瓶颈在cpu，gpu占27%
2workers
	worker_compute_score_by_query：450，性能瓶颈在cpu，gpu占59%
3workers
	worker_compute_score_by_query：590，性能瓶颈在cpu，gpu占85%
4workers
	worker_compute_score_by_query：700，性能瓶颈在cpu，gpu占99%
#### 同时加载bge-large-zh-v1.5和bge-reranker-base模型
同时请求：worker_embed_query和worker_compute_score_by_query
1workers：100，gpu占24%
2workers：180，gpu占59%
3workers：280，gpu占81%

### RTX 3090
#### 只加载bge-large-zh-v1.5模型
1workers
	worker_embed_query：100，性能瓶颈在cpu，gpu占16%
	worker_embed_documents：30，性能瓶颈在cpu，gpu占81%，gpu占用跟字数成正比
2workers
	worker_embed_query：200，性能瓶颈在cpu，gpu占70%
	worker_embed_documents：35，性能瓶颈在gpu，gpu占100%
3workers
	worker_embed_query：280，性能瓶颈在cpu，gpu占97%
	worker_embed_documents：30，性能瓶颈在gpu，gpu占100%
4workers
	worker_embed_query：350，gpu占100%

#### 只加载bge-reranker-base模型
1workers
	worker_compute_score_by_query：150，性能瓶颈在cpu，gpu占33%
2workers
	worker_compute_score_by_query：270，性能瓶颈在cpu，gpu占67%
3workers
	worker_compute_score_by_query：390，性能瓶颈在cpu，gpu占92%
4workers
	worker_compute_score_by_query：450，性能瓶颈在cpu，gpu占99%
#### 同时加载bge-large-zh-v1.5和bge-reranker-base模型
同时请求：worker_embed_query和worker_compute_score_by_query
1workers：70，gpu占30%
2workers：140，gpu占68%
3workers：200，gpu占90%
	
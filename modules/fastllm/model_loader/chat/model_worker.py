"""
A model worker that executes the model.
"""
import json
from typing import Optional

import torch
from transformers import set_seed

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.utils import get_context_length


class ModelWorker:
    def __init__(
            self,
            model_path: str,
            model_name: str,
            device: str,
            num_gpus: int,
            max_gpu_memory: Optional[str] = None,
            revision: str = None,
            dtype: Optional[torch.dtype] = None,
            load_8bit: bool = False,
            cpu_offloading: bool = False,
            gptq_config: Optional[GptqConfig] = None,
            awq_config: Optional[AWQConfig] = None,
            exllama_config: Optional[ExllamaConfig] = None,
            xft_config: Optional[XftConfig] = None,
            stream_interval: int = 2,
            embed_in_truncate: bool = False,
            seed: Optional[int] = None,
            debug: bool = False,
    ):

        print(f"Loading the model {model_name} ...")
        self.model, self.tokenizer = load_model(
            model_path,
            revision=revision,
            device=device,
            num_gpus=num_gpus,
            max_gpu_memory=max_gpu_memory,
            dtype=dtype,
            load_8bit=load_8bit,
            cpu_offloading=cpu_offloading,
            gptq_config=gptq_config,
            awq_config=awq_config,
            exllama_config=exllama_config,
            xft_config=xft_config,
            debug=debug,
        )
        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

    def generate_stream_gate(self, params):
        if self.device == "npu":
            import torch_npu

            torch_npu.npu.set_device("npu:0")

        try:
            if self.seed is not None:
                set_seed(self.seed)
            for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    params,
                    self.device,
                    self.context_len,
                    self.stream_interval,
            ):
                ret = {
                    "text": output["text"],
                    "error_code": 0,
                }
                if "usage" in output:
                    ret["usage"] = output["usage"]
                if "finish_reason" in output:
                    ret["finish_reason"] = output["finish_reason"]
                if "logprobs" in output:
                    ret["logprobs"] = output["logprobs"]
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
            yield json.dumps(ret).encode() + b"\0"
        except (ValueError, RuntimeError) as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_gate(self, params):
        for x in self.generate_stream_gate(params):
            pass
        return json.loads(x[:-1].decode())


if __name__ == "__main__":
    from modules.fastllm.model_loader.utils import detect_device

    worker = ModelWorker(
        model_path=r"E:\WorkSpace\LLMWorkSpace\Models\LLM\Qwen1.5-0.5B-Chat",
        model_name="Qwen1.5-0.5B-Chat",
        device=detect_device(),
        num_gpus=1,
        max_gpu_memory=None
    )

    params = {
        'model': 'Qwen1.5-0.5B-Chat',
        'prompt': '<|im_start|>system\nYou are a helpful '
                  'assistant.<|im_end|>\n<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n',
        'temperature': 0.7,
        'logprobs': None,
        'top_p': 1.0,
        'top_k': -1,
        'presence_penalty': 0.0,
        'frequency_penalty': 0.0,
        'max_new_tokens': 8171,
        'echo': False,
        'stop_token_ids': [151643, 151644, 151645],
        'stop': ['<|endoftext|>']
    }

    result = worker.generate_gate(params)
    print(result)

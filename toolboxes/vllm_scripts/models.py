"""
Centralized model execution profiles for B70 vLLM toolbox.

Multi-GPU (TP=2) support: models with valid_tp=[1,2] can use tensor parallelism
across 2+ Intel Arc GPUs. Small models (<~15B params) fit on a single B70 (32GB VRAM)
and only need TP=1. Larger models benefit from splitting across 2 GPUs for both
capacity (fit) and throughput (wider memory bandwidth).

TP selection rules:
  - valid_tp=[1]          -> single-GPU only (small models, or quantized models that fit)
  - valid_tp=[1,2]        -> user may select TP=1 or TP=2 depending on model size / VRAM
  - valid_tp=[1,2,...,N]  -> supports up to N GPUs
"""

GPU_UTIL = "0.90"
OFF_NUM_PROMPTS = 500 
OFF_FORCED_OUTPUT = "512"
DEFAULT_BATCH_TOKENS = "8192"

B70_ENV = {
    "VLLM_TARGET_DEVICE": "xpu",
    "VLLM_MLA_DISABLE": "1",
    "VLLM_USE_V1": "1",
    "VLLM_ENGINE_READY_TIMEOUT_S": "600",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn"
}

MODEL_TABLE = {
    # -- Small models: fit on 1x B70, TP=1 only --

    # 1. Llama 3.1 8B Instruct
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536",
        "language_model_only": True,
        "env": B70_ENV
    },

    # 2. Qwen 3.5 9B (Native FP16)
    "Qwen/Qwen3.5-9B": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536",
        "language_model_only": True,
        "env": B70_ENV
    },

    # -- Medium/large models: TP=1 or TP=2 (multi-GPU) --

    # 3. Qwen 3.6 27B GPTQ-4bit -- ~15GB, fits on 1x but benefits from TP=2
    "btbtyler09/Qwen3.6-27B-GPTQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.95",
        "env": B70_ENV
    },

    # 4. GLM 4.7 Flash 30B GPTQ -- ~17GB, fits on 1x but benefits from TP=2
    "FayeQuant/GLM-4.7-Flash-GPTQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "16",
        "max_tokens": "1024",
        "ctx": "1536",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.98",
        "env": B70_ENV
    },

    # 5. Qwen 3.6 35B-A3B (MoE) -- FP8 quantized, TP=2 on 2x B70, max context 65536
    "Qwen/Qwen3.6-35B-A3B": {
        "trust_remote": True,
        "valid_tp": [2],
        "max_num_seqs": "32",
        "max_tokens": "65536",
        "ctx": "65536",
        "language_model_only": True,
        "quantization": "fp8",
        "gpu_util": "0.90",
        "enforce_eager": True,
        "env": {
            "VLLM_TARGET_DEVICE": "xpu",
            "ZE_AFFINITY_MASK": "0,1",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True"
        }
    }
}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-9B",
    "btbtyler09/Qwen3.6-27B-GPTQ-4bit",
    "FayeQuant/GLM-4.7-Flash-GPTQ-4bit",
    "Qwen/Qwen3.6-35B-A3B"
]

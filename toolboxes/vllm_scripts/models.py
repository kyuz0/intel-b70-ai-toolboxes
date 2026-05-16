"""
Centralized model execution profiles for B70 vLLM toolbox.
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
    # 1. Llama 3.1 8B Instruct
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1],
        "max_num_seqs": "64",
        "max_tokens": "32768",
        "ctx": "65536",
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

    # 3. Qwen 3.5 27B GPTQ
    "btbtyler09/Qwen3.6-27B-GPTQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "32",
        "max_tokens": "16384",
        "ctx": "20480",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.95",
        "env": B70_ENV
    },

    # 4. GLM 4.7 Flash 30B GPTQ
    "FayeQuant/GLM-4.7-Flash-GPTQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "16",
        "max_tokens": "1024",
        "ctx": "1536",
        "language_model_only": True,
        "enforce_eager": True,
        "gpu_util": "0.98",
        "env": B70_ENV
    }
}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-9B",
    "btbtyler09/Qwen3.6-27B-GPTQ-4bit",
    "FayeQuant/GLM-4.7-Flash-GPTQ-4bit"
]

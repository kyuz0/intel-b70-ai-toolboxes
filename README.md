# Intel B70 Llama.cpp & vLLM Toolboxes

This project provides pre-built containers (“toolboxes”) for running LLMs on **Intel Arc B70** (and other modern Intel GPUs) using `llama.cpp` and `vLLM`. Toolbx is the standard developer container system in Fedora (and works on Ubuntu, openSUSE, Arch, etc).

📊 **Interactive Benchmarks:** Live performance results are available at [kyuz0.github.io/intel-b70-ai-toolboxes/](https://kyuz0.github.io/intel-b70-ai-toolboxes/).

---

## Qwen3.8 episode: run through Cockpit

Use [AI Toolbox Cockpit](https://github.com/kyuz0/ai-toolbox-cockpit) to install these images, download the compatible model/quantization and choose the one/two-B70 serving profile. The [B70 setup guide](https://github.com/kyuz0/ai-toolbox-cockpit/blob/main/docs/INTEL_B70.md) includes embedded and separate MTP heads, Strata preparation, and GPU selection.

| Public image | Episode configuration |
| --- | --- |
| `docker.io/kyuz0/intel-b70-vllm-toolbox:upstream` | Upstream vLLM0.31/NEO26.35: 27B RedHatAI INT4 on one/two B70s; official FP8 on two; MTP3 |
| `docker.io/kyuz0/intel-b70-vllm-toolbox:latest` | Intel LLM Scaler0.26.0-b2: 27B INT4 baseline on one B70 |
| `docker.io/kyuz0/intel-b70-vllm-toolbox:exl3-xpu` | Experimental EXL3 4.00bpw on one B70; MTP3; prefix caching off |
| `docker.io/kyuz0/intel-b70-vllm-toolbox:fp8-onecard` | Experimental Optimization Lab official FP8 on one B70; CPU embeddings/MTP5; prefix caching off |
| `docker.io/kyuz0/intel-b70-ai-toolboxes:sycl` | 27B UD-Q4_K_XL, baseline/MTP3, F16 KV, optimized weight path disabled |
| `docker.io/kyuz0/intel-b70-ai-toolboxes:vulkan` | 27B UD-Q4_K_XL, baseline/MTP3, F16 KV |
| `docker.io/kyuz0/intel-b70-ai-toolboxes:sycl-flash-next` | Patched Flash Next AtomicChat AD4.27 plus Q4 MTP on two B70s |
| `docker.io/kyuz0/intel-b70-ai-toolboxes:strata-sycl` | Strata Flash Next AtomicChat native pack plus official Q2 MTP on two B70s |

Each engine has a build workflow under [Actions](https://github.com/kyuz0/intel-b70-ai-toolboxes/actions). Flash Next, Strata and EXL3 pin their source revisions; the Optimization Lab image packages its pinned publisher binary and matching Python overlays. Its workflow performs the required manifest normalization before the container build. Standard llama.cpp builds accept a `llama_ref` input for reproducing an exact source revision and retain the existing rolling default.

## Table of Contents

- [Interactive Benchmarks](#interactive-benchmarks)
- [Supported Toolboxes](#supported-toolboxes)
- [Quick Start](#quick-start)
- [Host Configuration](#host-configuration)
- [Building Locally](#building-locally)

## Interactive Benchmarks

Interactive performance benchmark results for different backends, models, and quantization levels running on the Intel Arc B70 GPU are published at:

👉 **[Intel Arc B70 LLM Benchmarks](https://kyuz0.github.io/intel-b70-ai-toolboxes/)**

---

## Supported Toolboxes

You can check the containers on DockerHub: [kyuz0/intel-b70-ai-toolboxes](https://hub.docker.com/r/kyuz0/intel-b70-ai-toolboxes/tags).

| Container / Repo | Backend/Stack | Purpose / Notes |
| :--- | :--- | :--- |
| `kyuz0/intel-b70-ai-toolboxes:sycl` | Intel oneAPI SYCL | Native Intel backend for llama.cpp using Level Zero. Includes oneAPI and the GPU runtime; the Qwen3.8 27B profile disables the optimized weight path. |
| `kyuz0/intel-b70-ai-toolboxes:vulkan` | Vulkan (Mesa/Intel) | Universal backend for llama.cpp using Vulkan. Recommended for compatibility across different host setups and older Intel hardware. |
| `kyuz0/intel-b70-ai-toolboxes:openvino` | Intel OpenVINO | OpenVINO backend for llama.cpp. Translates GGML graphs into OpenVINO for Intel-optimized inference on CPUs and GPUs. Auto-configured to target the discrete GPU with stateful KV cache. |
| `kyuz0/intel-b70-vllm-toolbox:latest` | Intel vLLM Scaler | Official Intel vLLM stack optimized for Arc Pro B70, featuring an interactive TUI launcher (`start-vllm`). |

> The Llama.cpp containers are **automatically** rebuilt whenever the Llama.cpp master branch is updated. The vLLM container can be rebuilt using the provided GitHub action.

## Quick Start

Create and enter your toolbox of choice. **(Ubuntu users: remember to use `distrobox` instead of `toolbox` in the commands below).**

**Option A: Vulkan (Intel ANV)** - best for compatibility
```sh
toolbox create b70-llama-vulkan \
  --image docker.io/kyuz0/intel-b70-ai-toolboxes:vulkan \
  -- --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined

toolbox enter b70-llama-vulkan
```

**Option B: SYCL (Native Intel)** - best for performance
```sh
toolbox create b70-llama-sycl \
  --image docker.io/kyuz0/intel-b70-ai-toolboxes:sycl \
  -- --device /dev/dri --group-add video --group-add render --group-add sudo --security-opt seccomp=unconfined

toolbox enter b70-llama-sycl
```

**Option C: OpenVINO (Intel)** - graph-compiled Intel inference
```sh
toolbox create b70-llama-openvino \
  --image docker.io/kyuz0/intel-b70-ai-toolboxes:openvino \
  -- --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined

toolbox enter b70-llama-openvino
```

**Option D: vLLM (Intel Scaler)** - best for high-throughput serving
```sh
toolbox create b70-vllm \
  --image docker.io/kyuz0/intel-b70-vllm-toolbox:latest \
  -- --device /dev/dri --shm-size 200g --security-opt seccomp=unconfined --env no_proxy=localhost,127.0.0.1

toolbox enter b70-vllm
```

> **Tip:** You can also use the included `./refresh-toolboxes.sh [all|b70-llama-vulkan|b70-llama-sycl|b70-llama-openvino|b70-vllm]` script to automate the container pulling and creation process.

> **OpenVINO Notes:** Although an OpenVINO backend is available in the toolboxes, it is **not recommended** and is excluded from standard benchmarking. It fails to run with most modern models (especially Mixture-of-Experts/MoE architectures, which trigger OpenCL compilation crashes with `CL_OUT_OF_RESOURCES` errors on the GPU). If used, the container auto-exports `GGML_OPENVINO_DEVICE=GPU` and `GGML_OPENVINO_STATEFUL_EXECUTION=1` on entry. When benchmarking with `llama-bench`, you **must** pass `-fa 1` (flash attention) — this is an upstream requirement for the OpenVINO backend.
>
> ⚠️ **VRAM reporting:** `llama-cli --list-devices` will report **system RAM** (~64 GiB) instead of GPU VRAM (32 GiB). This is an upstream llama.cpp limitation — the OpenVINO backend does not query device VRAM. Use `gguf-vram-estimator.py` to check if a model fits in GPU memory before loading.

### 2. Check GPU Access
Inside the toolbox:
```sh
# For SYCL / vLLM
llama-cli --list-devices
# or
sycl-ls
```

### 3. Run Inference

**For Llama.cpp toolboxes:**
Download your GGUF models and run them natively.

*Server Mode (API):*
```sh
llama-server -m models/your-model.gguf -c 8192 -ngl 999
```

*CLI Mode:*
```sh
llama-cli -ngl 999 -m models/your-model.gguf -p "Write a haiku about Intel graphics."
```

**For vLLM toolbox:**
The vLLM toolbox comes with an interactive TUI. Simply run:
```sh
start-vllm
```

## Host Configuration

Ensure you are running an up-to-date kernel (6.8+) for the best Intel GPU driver support (`i915` or `xe` drivers).

For some advanced hardware scheduling features or to enable GuC/HuC firmware on older kernels, you may need to add the following to your GRUB boot parameters:
`i915.enable_guc=3` or `intel_iommu=on` depending on the hardware platform (not always necessary for B70 which uses `xe` out of the box in newer kernels).

## Memory Planning and VRAM Estimator

To estimate VRAM requirements for models (including context overhead), use the included tool:

```bash
gguf-vram-estimator.py models/my-model.gguf --contexts 32768
```

## Building Locally

You can build the containers yourself to customize packages or llama.cpp versions.
```bash
cd toolboxes
docker build -t llama-sycl -f Dockerfile.sycl .
```

# Intel B70 Llama.cpp & vLLM Toolboxes

This project provides pre-built containers (“toolboxes”) for running LLMs on **Intel Arc B70** (and other modern Intel GPUs) using `llama.cpp` and `vLLM`. Toolbx is the standard developer container system in Fedora (and works on Ubuntu, openSUSE, Arch, etc).

---

## Table of Contents

- [Supported Toolboxes](#supported-toolboxes)
- [Quick Start](#quick-start)
- [Host Configuration](#host-configuration)
- [Building Locally](#building-locally)

## Supported Toolboxes

You can check the containers on DockerHub: [kyuz0/intel-b70-ai-toolboxes](https://hub.docker.com/r/kyuz0/intel-b70-ai-toolboxes/tags).

| Container / Repo | Backend/Stack | Purpose / Notes |
| :--- | :--- | :--- |
| `kyuz0/intel-b70-ai-toolboxes:sycl` | Intel oneAPI SYCL | Native Intel backend for llama.cpp. Fastest generation performance, utilizes Level Zero. Requires Intel oneAPI Base Toolkit components installed inside the container. |
| `kyuz0/intel-b70-ai-toolboxes:vulkan` | Vulkan (Mesa/Intel) | Universal backend for llama.cpp using Vulkan. Recommended for compatibility across different host setups and older Intel hardware. |
| `kyuz0/intel-b70-vllm-toolbox:dev` | Intel vLLM Scaler | Official Intel vLLM stack optimized for Arc Pro B70, featuring an interactive TUI launcher (`start-vllm`). |

> The Llama.cpp containers are **automatically** rebuilt whenever the Llama.cpp master branch is updated. The vLLM container can be rebuilt using the provided GitHub action.

## Quick Start

Create and enter your toolbox of choice. **(Ubuntu users: remember to use `distrobox` instead of `toolbox` in the commands below).**

**Option A: Vulkan (Intel ANV)** - best for compatibility
```sh
toolbox create llama-vulkan \
  --image docker.io/kyuz0/intel-b70-ai-toolboxes:vulkan \
  -- --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined

toolbox enter llama-vulkan
```

**Option B: SYCL (Native Intel)** - best for performance
```sh
toolbox create llama-sycl \
  --image docker.io/kyuz0/intel-b70-ai-toolboxes:sycl \
  -- --device /dev/dri --group-add video --group-add render --group-add sudo --security-opt seccomp=unconfined

toolbox enter llama-sycl
```

**Option C: vLLM (Intel Scaler)** - best for high-throughput serving
```sh
toolbox create vllm \
  --image docker.io/kyuz0/intel-b70-vllm-toolbox:dev \
  -- --device /dev/dri --shm-size 200g --security-opt seccomp=unconfined --env no_proxy=localhost,127.0.0.1

toolbox enter vllm
```

> **Tip:** You can also use the included `./refresh-toolboxes.sh [all|llama-vulkan|llama-sycl|vllm]` script to automate the container pulling and creation process.

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

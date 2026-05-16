# AGENTS.md

## 1. Project Context
This repository contains the CI/CD pipeline and toolbox generation scripts for Intel Arc Pro B70 AI environments, supporting both `llama.cpp` and `vLLM` runtimes.

## 2. Environment Constraints (CRITICAL)
**DO NOT EXECUTE CODE.** You are running on a workstation with **NO GPU ACCESS**. 
- NEVER attempt to run `./refresh-toolboxes.sh`.
- NEVER attempt to use `toolbox` or `distrobox` commands.
- NEVER try to run or build the containers locally to "test" your work.
Your role is strictly to edit and manage the codebase. The user will handle all execution and testing on the actual hardware, so if you need to run these commands or any command inside a toolbox, ask the user and they will manually SSH into the box and run these commands for you.

## 3. Architecture & Project Layout
*Where to find things and how they are isolated.*
- **`toolboxes/`**: Contains the Dockerfiles for image generation.
  - `Dockerfile.sycl` and `Dockerfile.vulkan` (for llama.cpp).
  - `Dockerfile.llm-scaler-vllm` (for vLLM).
- **`toolboxes/vllm_scripts/`**: Python/Bash utilities injected exclusively into the vLLM container during its build (e.g., `start_vllm.py`, `models.py`). *Do not mix these with llama.cpp configurations.*
- **`.github/workflows/`**: The CI/CD pipelines.
  - `build_and_publish.yml` is strictly for rebuilding the llama.cpp images.
  - `build_vllm.yml` is strictly for rebuilding the vLLM image.

## 4. Technical Quirks & Boundaries (CRITICAL)
*Specific "gotchas" you must know to avoid breaking the environment.*
- **Toolbox/Distrobox Paradigm:** Never write standard `docker run` commands with rigid volume mounts (e.g., `-v /home/user/.cache:/root/.cache`). The toolbox/distrobox environment natively maps the user's home directory. Stick to this paradigm.
- **Hardware Bindings:** Any new container initialization command must explicitly pass the GPU: `--device /dev/dri`.
- **vLLM Specific Constraints:** 
  - The vLLM container requires large shared memory: `--shm-size 200g`.
  - It also requires `--security-opt seccomp=unconfined`.
  - Python scripts running inside vLLM *must* pass `VLLM_TARGET_DEVICE=xpu` in their environment variables.
- **Separation of Concerns:** Never mix or consolidate `llama.cpp` logic with `vLLM` logic. Maintain strict isolation between their workflows, Dockerfiles, and injected scripts to avoid dependency conflicts.

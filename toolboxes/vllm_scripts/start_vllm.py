#!/usr/bin/env python3
import sys
import os
import tempfile
import subprocess
import time
from pathlib import Path
import shutil

# Add directory to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

try:
    from models import MODEL_TABLE, MODELS_TO_RUN, GPU_UTIL
except ImportError:
    print("Error: Could not import models.py config. Ensure models.py is in the same directory.")
    sys.exit(1)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")

def check_dependencies():
    if not shutil.which("dialog"):
        print("Error: 'dialog' is required. Please install it (apt-get install dialog).")
        sys.exit(1)

def detect_gpus():
    try:
        return len(list(Path("/dev/dri").glob("renderD*")))
    except:
        return 1

def run_dialog(args):
    """Runs dialog and returns stderr (selection)."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled

def nuke_vllm_cache():
    """Removes vLLM cache directory."""
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            print(f"Clearing vLLM cache at {cache}...", end="", flush=True)
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            print(" Done.")
            time.sleep(1)
        except Exception as e:
            print(f" Failed: {e}")

def configure_and_launch(model_idx, gpu_count):
    model_id = MODELS_TO_RUN[model_idx]
    config = MODEL_TABLE[model_id]
    
    # Static Config Setup
    valid_tps = config.get("valid_tp", [1])
    max_tp = max(valid_tps) if valid_tps else 1
    
    # Defaults
    current_tp = min(gpu_count, max_tp)
    current_seqs = int(config.get("max_num_seqs", "128"))
    current_ctx = int(config.get("ctx", "4096"))
    current_util = float(config.get("gpu_util", GPU_UTIL))
    use_eager = config.get("enforce_eager", True)
    clear_cache = False
    
    name = model_id.split("/")[-1]
    
    while True:
        cache_status = "YES" if clear_cache else "NO"
        eager_status = "YES" if use_eager else "NO"
        
        menu_args = [
            "--clear", "--backtitle", f"B70 vLLM Launcher (GPUs: {gpu_count} detected)",
            "--title", f"Configuration: {name}",
            "--menu", "Customize Launch Parameters:", "22", "65", "9",
            "1", f"Tensor Parallelism:   {current_tp}",
            "2", f"Concurrent Requests:  {current_seqs}",
            "3", f"Context Length:       {current_ctx}",
            "4", f"GPU Utilization:      {current_util}",
            "5", f"Erase vLLM Cache:     {cache_status}",
            "6", f"Force Eager Mode:     {eager_status}",
            "7", "LAUNCH SERVER"
        ]
        
        choice = run_dialog(menu_args)
        if not choice: return False # Back/Cancel
        
        if choice == "1":
            new_tp = run_dialog(["--title", "Tensor Parallelism", "--rangebox", f"Set TP Size (1-{max_tp})", "10", "40", "1", str(max_tp), str(current_tp)])
            if new_tp: current_tp = int(new_tp)
        elif choice == "2":
            new_seqs = run_dialog(["--title", "Concurrent Requests", "--inputbox", "Max Concurrent Requests:", "10", "40", str(current_seqs)])
            if new_seqs: current_seqs = int(new_seqs)
        elif choice == "3":
            new_ctx = run_dialog(["--title", "Context Length", "--inputbox", "Max Model Context Length:", "10", "40", str(current_ctx)])
            if new_ctx: current_ctx = int(new_ctx)
        elif choice == "4":
            new_util = run_dialog(["--title", "GPU Utilization", "--inputbox", "GPU Memory Utilization (0.1 - 1.0):", "10", "40", str(current_util)])
            if new_util: current_util = float(new_util)
        elif choice == "5":
            clear_cache = not clear_cache
        elif choice == "6":
            use_eager = not use_eager
        elif choice == "7":
            break
            
    # Build Command
    subprocess.run(["clear"])
    if clear_cache:
        nuke_vllm_cache()
    
    cmd = [
        "vllm", "serve", model_id,
        "--served-model-name", name,
        "--host", HOST,
        "--port", PORT,
        "--tensor-parallel-size", str(current_tp),
        "--max-num-seqs", str(current_seqs),
        "--max-model-len", str(current_ctx),
        "--gpu-memory-utilization", str(current_util),
        "--dtype", "bfloat16",
        "--block-size", "64",
        "--disable-log-requests",
        "--no-enable-prefix-caching"
    ]
    
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    if use_eager: cmd.append("--enforce-eager")
    
    # Env Vars
    env = os.environ.copy()
    env.update(config.get("env", {}))
    
    print("\n" + "="*60)
    print(f" Launching: {name}")
    print(f" Config:    TP={current_tp} | Seqs={current_seqs} | Ctx={current_ctx} | Util={current_util} | Eager={use_eager}")
    if clear_cache:
        print(f" Action:    Clearing vLLM Cache (~/.cache/vllm)")
    print(f" Command:   {' '.join(cmd)}")
    print("="*60 + "\n")
    
    os.execvpe("vllm", cmd, env)

def main():
    check_dependencies()
    gpu_count = detect_gpus()
    
    while True:
        menu_items = []
        for i, m_id in enumerate(MODELS_TO_RUN):
            name = m_id.split("/")[-1]
            menu_items.extend([str(i), name])
            
        choice = run_dialog([
            "--clear", "--backtitle", f"B70 vLLM Launcher (GPUs: {gpu_count})",
            "--title", "Select Model",
            "--menu", "Choose a model to serve:", "15", "60", "8"
        ] + menu_items)
        
        if not choice:
            subprocess.run(["clear"])
            print("Selection cancelled.")
            sys.exit(0)
            
        configure_and_launch(int(choice), gpu_count)

if __name__ == "__main__":
    main()

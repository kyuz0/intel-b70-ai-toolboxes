#!/usr/bin/env python3
import subprocess, time, json, sys, os, requests, argparse, shutil, tempfile
from pathlib import Path

# Add directory to path to import config
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

try:
    import models
except ImportError:
    print("Error: Could not import models.py config. Ensure models.py is in the same directory.")
    sys.exit(1)

MODEL_TABLE = models.MODEL_TABLE
MODELS_TO_RUN = models.MODELS_TO_RUN
GPU_UTIL = models.GPU_UTIL
OFF_NUM_PROMPTS = models.OFF_NUM_PROMPTS
OFF_FORCED_OUTPUT = models.OFF_FORCED_OUTPUT
DEFAULT_BATCH_TOKENS = models.DEFAULT_BATCH_TOKENS

# Fallbacks
FALLBACK_INPUT_LEN  = 1024
FALLBACK_OUTPUT_LEN = 512

RESULTS_DIR = Path("~/vllm_benchmark_results_b70").expanduser()
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def run_dialog(args):
    """Runs dialog and returns stderr (selection line). Returns None if user cancelled."""
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            subprocess.run(cmd, stderr=tf, check=True)
            tf.seek(0)
            return tf.read().strip()
        except subprocess.CalledProcessError:
            return None # User cancelled/pressed ESC

def log(msg): print(f"\n[BENCH] {msg}")

def get_gpu_count():
    try:
        count = 0
        for path in Path("/sys/class/drm").glob("renderD*/device/vendor"):
            if "0x8086" in path.read_text(): count += 1
        return count if count > 0 else 1
    except:
        return 1

def kill_vllm():
    cmds = [
        "pgrep -f 'vllm bench' | xargs -r kill -9",
        "pgrep -f 'vllm serve' | xargs -r kill -9",
        "pgrep -f 'VLLM::' | xargs -r kill -9",
        "pgrep -f 'ray::' | xargs -r kill -9"
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def nuke_vllm_cache():
    cache = Path.home() / ".cache" / "vllm"
    if cache.exists():
        try:
            subprocess.run(["rm", "-rf", str(cache)], check=True)
            cache.mkdir(parents=True, exist_ok=True)
            time.sleep(2)
        except: pass

def get_dataset():
    cache_dir = Path.home() / ".cache" / "sharegpt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "ShareGPT_V3_unfiltered_cleaned_split.json"
    if data_path.exists():
        if data_path.stat().st_size > 100_000_000: # ~540MB expected
            return str(data_path)
        else:
            log("Found corrupted/incomplete ShareGPT dataset. Re-downloading...")
            data_path.unlink()
    
    log("Downloading ShareGPT dataset...")
    url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        tmp_path = data_path.with_suffix(".tmp")
        with open(tmp_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        tmp_path.rename(data_path)
        return str(data_path)
    except Exception as e:
        log(f"WARNING: ShareGPT download failed ({e}). using RANDOM.")
        return None

def get_model_args(model, tp_size, overrides=None):
    config = MODEL_TABLE.get(model, {})
    overrides = overrides or {}
    
    util = overrides.get("gpu_util", config.get("gpu_util", GPU_UTIL))
    max_seq_override = overrides.get("max_num_seqs", config.get("max_num_seqs", "128"))

    cmd = [
        "--model", model,
        "--gpu-memory-utilization", str(util),
        "--dtype", "bfloat16",
        "--block-size", "64",
        "--tensor-parallel-size", str(tp_size),
        "--max-num-seqs", str(max_seq_override),
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--disable-custom-all-reduce"
    ]
    
    ctx = overrides.get("ctx", config.get("ctx"))
    if ctx:
        cmd.extend(["--max-model-len", str(ctx)])
        
    kv_cache_dtype = overrides.get("kv_cache_dtype", config.get("kv_cache_dtype"))
    if kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", kv_cache_dtype])
        
    if config.get("trust_remote"): cmd.append("--trust-remote-code")
    use_eager = overrides.get("enforce_eager", config.get("enforce_eager", False))
    if use_eager: cmd.append("--enforce-eager")
    if config.get("language_model_only"): cmd.extend(["--limit-mm-per-prompt", '{"image": 0, "video": 0}'])
    
    return cmd

def run_throughput(model, tp_size, output_dir=RESULTS_DIR, overrides=None):
    if tp_size not in MODEL_TABLE[model].get("valid_tp", [1]): return
    overrides = overrides or {}
    
    model_safe = model.replace("/", "_")
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    tag = overrides.get("tag", "").strip()
    tag_suffix = f"_{tag}" if tag else ""
    output_file = output_dir_path / f"{model_safe}_tp{tp_size}{tag_suffix}_throughput.json"
    
    if output_file.exists():
        log(f"SKIP {model} (TP={tp_size})")
        return

    dataset_path = get_dataset()
    dataset_args = ["--dataset-name", "sharegpt", "--dataset-path", dataset_path] if dataset_path else ["--input-len", "1024"]
    
    batch_tokens = str(overrides.get("max_tokens", MODEL_TABLE[model].get("max_tokens", DEFAULT_BATCH_TOKENS)))

    log(f"START {model} (TP={tp_size}) [Batch: {batch_tokens}]...")
    kill_vllm()
    nuke_vllm_cache()

    vllm_path = shutil.which("vllm") or "vllm"
    cmd = ["python3", "-W", "ignore", vllm_path, "bench", "throughput"] + get_model_args(model, tp_size, overrides)
    cmd.extend([
        "--num-prompts", str(OFF_NUM_PROMPTS),
        "--max-num-batched-tokens", batch_tokens,
        "--output-len", OFF_FORCED_OUTPUT,
        "--output-json", str(output_file),
        "--disable-log-stats"
    ])
    cmd.extend(dataset_args)

    env = os.environ.copy()
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    
    model_env = MODEL_TABLE[model].get("env", {})
    env.update(model_env)

    try: 
        subprocess.run(cmd, check=True, env=env)
    except Exception as e: 
        log(f"ERROR: Failed {model}")
        try:
            with open(output_file, 'w') as f:
                json.dump({"error": "Failed"}, f)
        except: pass

def print_summary(tps):
    print(f"\n{'MODEL':<40} | {'TP':<2} | {'Tag':<15} | {'Tok/s':<8}")
    print("-" * 75)
    
    for m in MODELS_TO_RUN:
        msafe = m.replace("/", "_")
        name_cell = m.split('/')[-1]
        
        for tp in tps:
            if tp not in MODEL_TABLE[m].get("valid_tp", [1]): continue
            
            prefix = f"{msafe}_tp{tp}"
            
            tags = set()
            for p in RESULTS_DIR.glob(f"{prefix}*_throughput.json"):
                name_part = p.name[len(prefix):-len("_throughput.json")]
                tag = name_part.lstrip("_")
                tags.add(tag)
                
            if not tags:
                tags.add("")
                
            for tag in sorted(list(tags)):
                tag_suffix = f"_{tag}" if tag else ""
                
                try: 
                    p1 = RESULTS_DIR / f"{prefix}{tag_suffix}_throughput.json"
                    if p1.exists():
                        d1 = json.loads(p1.read_text())
                        val1 = d1["error"] if "error" in d1 else f"{d1.get('tokens_per_second', 0):.1f}"
                    else:
                        val1 = "N/A"
                except: val1 = "N/A"

                display_tag = tag if tag else "(Default)"
                print(f"{name_cell:<40} | {tp:<2} | {display_tag:<15} | {val1:<8}")
                
    print("-" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLLM High-Concurrency Throughput Benchmark Suite for B70")
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    parser.add_argument("--tui", action="store_true", help="Launch interactive configuration UI")
    args = parser.parse_args()
    
    gpu_count = get_gpu_count()
    log(f"Detected {gpu_count} Intel GPU(s)")
    log("NOTE: Running Peak Throughput Benchmark. This simulates high-concurrency batching to saturate hardware bandwidth.")
    
    valid_tp_args = [t for t in args.tp if t <= gpu_count]
    if not valid_tp_args:
        log(f"Requested TP={args.tp} but only {gpu_count} GPU(s) detected. Nothing to run.")
        sys.exit(0)

    selected_models = MODELS_TO_RUN
    
    if args.tui:
        if not shutil.which("dialog"):
            log("Error: 'dialog' is required for TUI. Please install it.")
            sys.exit(1)

        checklist_args = [
            "--clear", "--backtitle", "Intel B70 vLLM Benchmark Launcher",
            "--title", "Model Selection",
            "--checklist", "Select models to benchmark:", "20", "65", "10"
        ]
        
        for m in MODELS_TO_RUN:
            m_name = m.split("/")[-1]
            checklist_args.extend([m, m_name, "on"])
            
        choice = run_dialog(checklist_args)
        
        if choice is None:
            subprocess.run(["clear"])
            print("Cancelled by user.")
            sys.exit(0)
            
        import shlex
        selected_models = [m for m in shlex.split(choice)]
        
        if not selected_models:
            subprocess.run(["clear"])
            print("No models selected. Exiting.")
            sys.exit(0)

    kill_vllm()
    for tp in valid_tp_args:
        for m in selected_models:
            overrides = {}
            if args.tui:
                config = MODEL_TABLE.get(m, {})
                default_seqs = config.get("max_num_seqs", "128")
                default_tokens = config.get("max_tokens", DEFAULT_BATCH_TOKENS)
                default_util = config.get("gpu_util", GPU_UTIL)
                default_ctx = config.get("ctx", "4096")
                
                form_args = [
                    "--clear", "--backtitle", f"Intel B70 vLLM Benchmark Configuration (TP: {tp})",
                    "--title", f"Tune Parameters: {m.split('/')[-1]}",
                    "--form", "Edit the options below. Leave tag empty for no suffix.",
                    "15", "70", "5",
                    "Max Concurrent Seqs:", "1", "1",  str(default_seqs), "1", "25", "15", "0",
                    "Max Batched Tokens:", "2", "1", str(default_tokens), "2", "25", "15", "0",
                    "GPU Utilization (0-1):", "3", "1", str(default_util), "3", "25", "15", "0",
                    "Max Context Length:", "4", "1", str(default_ctx), "4", "25", "15", "0",
                    "Filename Tag (Optional):", "5", "1", "", "5", "25", "15", "0"
                ]
                
                form_res = run_dialog(form_args)
                if form_res is None:
                    subprocess.run(["clear"])
                    print(f"Skipping {m} (TP={tp}) due to user cancellation.")
                    continue
                    
                lines = form_res.splitlines()
                if len(lines) >= 5:
                    overrides["max_num_seqs"] = lines[0].strip()
                    overrides["max_tokens"] = lines[1].strip()
                    overrides["gpu_util"] = lines[2].strip()
                    
                    ctx_val = lines[3].strip()
                    if ctx_val and ctx_val.lower() != "auto":
                        overrides["ctx"] = ctx_val
                        
                    overrides["tag"] = lines[4].strip()
            
            run_throughput(m, tp, RESULTS_DIR, overrides=overrides)
            
    print_summary(valid_tp_args)

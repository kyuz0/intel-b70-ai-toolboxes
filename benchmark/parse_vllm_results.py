#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path

# Config
BENCHMARK_DIR = Path("vllm-benchs")
OUTPUT_FILE = Path("../docs/vllm-results.json")

# Regex to parse model name for quantization and parameters
PARAMS_REGEX = r"(\d+(?:\.\d+)?)B"
QUANT_REGEX = r"(FP8|AWQ|GPTQ|BF16|4bit|Int4)"

def extract_meta(model_name):
    # Params
    params_match = re.search(PARAMS_REGEX, model_name, re.IGNORECASE)
    params_b = float(params_match.group(1)) if params_match else None
    
    # Quant
    quant_match = re.search(QUANT_REGEX, model_name, re.IGNORECASE)
    quant = quant_match.group(1).upper() if quant_match else "BF16"
    if quant in ["4BIT", "INT4"]:
        if "GPTQ" in model_name: quant = "GPTQ-4bit"
        elif "AWQ" in model_name: quant = "AWQ-4bit"
        else: quant = "4-bit"

    return params_b, quant

def parse_logs():
    runs = []
    
    if not BENCHMARK_DIR.exists():
        print(f"Warning: {BENCHMARK_DIR} does not exist. Please create it and drop your benchmark JSONs there.")
        return runs

    print(f"Scanning {BENCHMARK_DIR}...")
    files = list(BENCHMARK_DIR.glob("*.json"))
    
    for f in files:
        fname = f.name
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"Skipping bad JSON: {fname} ({e})")
            continue

        # Infer metadata from filename
        parts = fname.split("_tp")
        if len(parts) < 2: continue
        
        model_part = parts[0]
        rest = parts[1] 
        
        # TP
        tp_match = re.match(r"^(\d+)", rest)
        if not tp_match: continue
        tp = int(tp_match.group(1))
        
        env = f"TP{tp}"
        
        # Model Name Restoration
        if "_" in model_part:
            model_display = model_part.replace("_", "/", 1)
        else:
            model_display = model_part
            
        params_b, quant = extract_meta(model_display)
        
        base_run = {
            "model": model_display,
            "model_clean": model_display,
            "env": env,
            "variant": "intel_b70",
            "gpu_config": "dual" if tp > 1 else "single",
            "quant": quant,
            "params_b": params_b,
            "name_params_b": params_b,
            "backend": "vLLM", 
            "error": False
        }

        if "throughput" in fname:
            tps = data.get("tokens_per_second", 0)
            run = base_run.copy()
            run["test"] = "Throughput"
            run["tps_mean"] = tps
            if tps == 0 and "error" in str(data).lower():
                run["error"] = True
            runs.append(run)
            
        # Add parsing for latency if we generate it in the future
        elif "latency" in fname:
            pass

    return runs

if __name__ == "__main__":
    runs = parse_logs()
    
    if not runs:
        print("No valid runs found to parse.")
        # Ensure directory exists for output even if empty
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"runs": []}, f, indent=2)
        print(f"Written empty schema to {OUTPUT_FILE}")
        exit(0)

    data = {"runs": runs}
    runs_count = len(runs)
    print(f"Parsed {runs_count} runs.")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Written to {OUTPUT_FILE}")

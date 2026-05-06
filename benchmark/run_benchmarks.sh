#!/usr/bin/env bash
set -uo pipefail

MODEL_DIR="$(realpath ~/models)"
RESULTDIR="$(realpath ~/llamacpp-bench-b70)"
mkdir -p "$RESULTDIR"

RUN_64K=0
if [[ "${1:-}" == "--64k" ]]; then
  RUN_64K=1
fi

# Capture system info
if [[ ! -f "$RESULTDIR/system_info.json" ]]; then
    python3 -c '
import platform, json, datetime
def get_distro():
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip("\"")
    except:
        return "Linux"
    return "Linux"

def get_linux_firmware():
    try:
        import subprocess
        result = subprocess.run(["rpm", "-q", "linux-firmware"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "unknown"

info = {
    "distro": get_distro(),
    "kernel": platform.release(),
    "linux_firmware": get_linux_firmware(),
    "timestamp": datetime.datetime.now().strftime("%d %b %Y")
}
print(json.dumps(info))
' > "$RESULTDIR/system_info.json"
    echo "Captured system info to $RESULTDIR/system_info.json"
fi

# Pick exactly one .gguf per model: either
#  - any .gguf without "-000*-of-" (single-file models)
#  - or the first shard "*-00001-of-*.gguf"
mapfile -t MODEL_PATHS < <(
  find "$MODEL_DIR" -type f -name '*.gguf' \
    \( -name '*-00001-of-*.gguf' -o -not -name '*-000*-of-*.gguf' \) \
    | sort
)

if (( ${#MODEL_PATHS[@]} == 0 )); then
  echo "❌ No models found under $MODEL_DIR – check your paths/patterns!"
  exit 1
fi

echo "Found ${#MODEL_PATHS[@]} model(s) to bench:"
for p in "${MODEL_PATHS[@]}"; do
  echo "  • $p"
done
echo

declare -A CMDS=(
  [sycl]="toolbox run -c llama-sycl -- env ONEAPI_DEVICE_SELECTOR=level_zero:gpu /usr/local/bin/llama-bench"
  [vulkan]="toolbox run -c llama-vulkan -- env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json /usr/local/bin/llama-bench"
)

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH" .gguf)"

  for ENV in "${!CMDS[@]}"; do
    CMD_EFFECTIVE="${CMDS[$ENV]}"

    # run twice: baseline and with flash attention
    for FA in 1; do
      SUFFIX=""
      EXTRA_ARGS=()
      if (( FA == 1 )); then
        SUFFIX="__fa1"
        EXTRA_ARGS=( -fa 1 )
      fi

      CTX_LIST=( default longctx16384 )
      if (( RUN_64K == 1 )); then
        CTX_LIST+=( longctx65536 )
      fi

      for CTX in "${CTX_LIST[@]}"; do
        CTX_SUFFIX=""
        CTX_ARGS=()
        if [[ "$CTX" == longctx16384 ]]; then
          CTX_SUFFIX="__longctx16384"
          CTX_ARGS=( -p 2048 -n 32 -d 16384 -ub 512 )
        elif [[ "$CTX" == longctx65536 ]]; then
          CTX_SUFFIX="__longctx65536"
          CTX_ARGS=( -p 2048 -n 32 -d 65536 -ub 512 )
        fi

        OUT="$RESULTDIR/${MODEL_NAME}__${ENV}${SUFFIX}${CTX_SUFFIX}.log"
        CTX_REPS=5
        if [[ "$CTX" == longctx16384 ]] || [[ "$CTX" == longctx65536 ]]; then
          CTX_REPS=3
        fi

        # VRAM Gatekeeper
        CTX_NUM=512
        if [[ "$CTX" == longctx16384 ]]; then CTX_NUM=16384; fi
        if [[ "$CTX" == longctx32768 ]]; then CTX_NUM=32768; fi
        if [[ "$CTX" == longctx65536 ]]; then CTX_NUM=65536; fi
        
        EST_GB=$(toolbox run -c llama-sycl -- /usr/local/bin/gguf-vram-estimator.py "$MODEL_PATH" -c $CTX_NUM 2>/dev/null | awk -F '|' '/^[ \t]*[0-9,]+[ \t]*\|/ {print $3}' | awk '{print $1}' | head -n1)
        if [[ -n "$EST_GB" ]]; then
          if (( $(awk -v est="$EST_GB" 'BEGIN {print (est > 31.5) ? 1 : 0}') )); then
            echo "⏭️  Skipping [${ENV}] ${MODEL_NAME}${SUFFIX}${CTX_SUFFIX:+ ($CTX_SUFFIX)}: Est VRAM ${EST_GB} GiB exceeds Intel Arc B70 limits (31.5 GiB max)"
            continue
          fi
        fi

        if [[ -s "$OUT" ]]; then
          echo "⏩ Skipping [${ENV}] ${MODEL_NAME}${SUFFIX}${CTX_SUFFIX:+ ($CTX_SUFFIX)}, log already exists at $OUT"
          continue
        fi

        FULL_CMD=( $CMD_EFFECTIVE -ngl 99 -mmp 0 -m "$MODEL_PATH" "${EXTRA_ARGS[@]}" "${CTX_ARGS[@]}" -r "$CTX_REPS" )

        printf "\n▶ [%s] %s%s%s\n" "$ENV" "$MODEL_NAME" "${SUFFIX:+ $SUFFIX}" "${CTX_SUFFIX:+ $CTX_SUFFIX}"
        printf "  → log: %s\n" "$OUT"
        printf "  → cmd: %s\n\n" "${FULL_CMD[*]}"

        if ! "${FULL_CMD[@]}" < /dev/null >"$OUT" 2>&1; then
          status=$?
          echo "✖ ! [${ENV}] ${MODEL_NAME}${SUFFIX}${CTX_SUFFIX:+ $CTX_SUFFIX} failed (exit ${status})" >>"$OUT"
          echo "  * [${ENV}] ${MODEL_NAME}${SUFFIX}${CTX_SUFFIX:+ $CTX_SUFFIX} : FAILED"
        fi
      done
    done
  done
done

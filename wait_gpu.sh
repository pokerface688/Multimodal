#!/usr/bin/env bash
# 轮询 GPU 0–3，首张当前无计算进程（无占用中的训练/推理进程）的卡上运行 main.py。
# 用法：在 Multimodal 目录下
#   ./wait_gpu.sh -d weather
#   ./wait_gpu.sh --dataset weather danmaku
# 不传 -d 时默认数据集为 weather。

set -euo pipefail

INTERVAL_SEC="${INTERVAL_SEC:-30}"
GPUS="${GPUS:-0 1 2 3}"

DATASETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: $0: -d/--dataset 需要至少一个数据集名" >&2
        exit 1
      fi
      while [[ $# -gt 0 && $1 != -* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    -h|--help)
      echo "Usage: $0 [-d|--dataset NAME [NAME ...]]"
      echo "  未指定 -d 时默认: weather"
      echo "  环境变量: INTERVAL_SEC, GPUS"
      exit 0
      ;;
    *)
      echo "error: 未知参数: $1" >&2
      echo "Usage: $0 [-d|--dataset NAME [NAME ...]]" >&2
      exit 1
      ;;
  esac
done
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(weather)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

count_compute_pids() {
  local gpu="$1"
  # 仅统计该 GPU 上的计算进程行（无进程时为空）
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^\s*$/d' | wc -l
}

while true; do
  for gpu in $GPUS; do
    n="$(count_compute_pids "$gpu")"
    if [[ "$n" -eq 0 ]]; then
      echo "[$(date -Iseconds)] GPU ${gpu} 无计算进程，启动任务（CUDA_VISIBLE_DEVICES=${gpu}，数据集: ${DATASETS[*]}）"
      exec env CUDA_VISIBLE_DEVICES="$gpu" python main.py \
        -d "${DATASETS[@]}" \
        --model_name deepseek-7b \
        --seed 39 \
        --batch_size 16 \
        --epoch 50 \
        --patience 10 \
        --use_text
    fi
  done
  echo "[$(date -Iseconds)] 0–3 均有进程占用，${INTERVAL_SEC}s 后重试（可 export INTERVAL_SEC=60 改间隔）"
  sleep "$INTERVAL_SEC"
done

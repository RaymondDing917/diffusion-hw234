#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_ROOT="${RUN_ROOT:-/root/shared-nvme/hw3}"
if [[ "$RUN_ROOT" == /root/shared-nvme/* && ! -d /root/shared-nvme ]]; then
    echo "Persistent storage /root/shared-nvme is not mounted" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"
export HF_HOME="${HF_HOME:-$RUN_ROOT/huggingface_cache}"
export WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$HF_HOME" "$WANDB_DIR"

run_training() {
    local name="$1"
    local variant="$2"
    shift 2
    local out_dir="$RUN_ROOT/$name"
    mkdir -p "$out_dir"
    if [[ -f "$out_dir/results.json" ]]; then
        echo "Already completed: $name"
        return
    fi

    local init_from="gpt2-medium"
    if [[ -f "$out_dir/ckpt.pt" ]]; then
        init_from="resume"
    fi
    echo "Running $name (initialization: $init_from; prompt: $variant)"
    HW3_PROMPT_VARIANT="$variant" python -u train.py \
        --init_from="$init_from" --out_dir="$out_dir" "$@" \
        2>&1 | tee -a "$out_dir/train.log"
}

run_baseline() {
    local out_dir="$RUN_ROOT/baseline"
    mkdir -p "$out_dir"
    if [[ -f "$out_dir/results.json" ]]; then
        echo "Already completed: baseline"
        return
    fi
    HW3_PROMPT_VARIANT=original python -u generate.py \
        --init_from=gpt2-medium --out_dir="$out_dir" \
        2>&1 | tee -a "$out_dir/generate.log"
}

run_mode() {
    case "$1" in
        baseline) run_baseline ;;
        default) run_training r128_a512 original ;;
        r16_a64) run_training r16_a64 original --lora_rank=16 --lora_alpha=64 ;;
        r16_a256) run_training r16_a256 original --lora_rank=16 --lora_alpha=256 ;;
        r196_a784) run_training r196_a784 original --lora_rank=196 --lora_alpha=784 ;;
        full) run_training full_finetune original --lora_rank=0 --dropout=0.05 ;;
        prompt_short) run_training r128_a512_short short ;;
        *) echo "Usage: bash run_hw3.sh {baseline|default|r16_a64|r16_a256|r196_a784|full|prompt_short|all}" >&2; exit 2 ;;
    esac
}

mode="${1:-default}"
if [[ "$mode" == all ]]; then
    for experiment in baseline default r16_a64 r16_a256 r196_a784 full prompt_short; do
        run_mode "$experiment"
    done
else
    run_mode "$mode"
fi

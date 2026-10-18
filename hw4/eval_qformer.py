import os
import argparse
import json
import random
from typing import Dict, List, Tuple, Optional
import yaml
import numpy as np
import torch
import torch.nn as nn
import gc
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from dit import DiT_Llama
from ddpm import ddpm_schedules
from image_caption_data import prompts_to_padded_hidden_states
from train_qformer import run_qformer_inference
from attention_visualization import setup_attention_capture, perform_attention_visualization

def eval_cifar10_qformer(args):
    device = torch.device(args.device)
    with open(args.config_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "Image_Generation_Configs" in data:
        cfg_list = data["Image_Generation_Configs"]
    elif isinstance(data, list):
        cfg_list = data
    else:
        raise ValueError("Config YAML must be a dict with key 'Image_Generation_Configs' or a list")

    channels = 3
    num_classes = 10
    image_size = 32

    for entry in cfg_list:
        if not isinstance(entry, dict):
            raise ValueError("Each config entry must be a mapping/dictionary")
        
        # Hyperparameters / overrides
        model_path = entry.get("model_path", None)
        if model_path is None or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        cfg_scale = float(entry.get("cfg", 1.0))
        save_path = entry.get("save_path", "./contents_qformer")
        num_query_tokens = int(entry.get("num_query_tokens", 4))
        transformer_hidden_size = int(entry.get("transformer_hidden_size", 768))
        n_T = int(entry.get("n_T", 1000))
        beta1 = float(entry.get("beta1", 1e-4))
        beta2 = float(entry.get("beta2", 2e-2))
        gpt2_layer_index = int(entry.get("gpt2_layer_index", 12))
        gpt2_cache_dir = entry.get("gpt2_cache_dir", "./data/gpt2")

        prompts = entry.get("prompts")
        classes = entry.get("classes")

        assert not (classes is None and prompts is None), "Must provide either classes or prompts"
        assert not (classes is not None and prompts is not None), "Must provide either classes or prompts, not both"

        # Build model per entry
        model = DiT_Llama(
            channels,
            image_size,
            dim=256,
            n_layers=10,
            n_heads=8,
            num_classes=num_classes,
            use_text_conditioning=classes is None,
            transformer_hidden_size=transformer_hidden_size,
            num_query_tokens=num_query_tokens,
        ).to(device)

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        # Scheduler per entry
        sched = ddpm_schedules(beta1, beta2, n_T)
        for k in list(sched.keys()):
            sched[k] = sched[k].to(device)

        if classes is not None:
            cond_classes = torch.tensor(classes, device=device, dtype=torch.long)
            text_states = None
        else:
            cond_classes = None
        # Optional token labels for x-axis when prompts provided
        token_labels_for_first = None
        if prompts is not None:
            # Tokenizer/LM per entry
            tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=gpt2_cache_dir)
            gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=gpt2_cache_dir).to(device)
            gpt2.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            text_states, text_masks = prompts_to_padded_hidden_states(prompts, gpt2, tokenizer, gpt2_layer_index, device)
            # Derive token labels from the first prompt for visualization
            try:
                enc = tokenizer(prompts[0], add_special_tokens=False)
                ids = enc.get("input_ids", [])
                if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], int):
                    token_labels_for_first = tokenizer.convert_ids_to_tokens(ids)
            except Exception:
                token_labels_for_first = None


        if 'attn_save_dir' in entry:
            attn_ctrl = setup_attention_capture(entry, model, token_labels_for_first)
        
        x_i = run_qformer_inference(
            model, sched, cond_classes, text_states, text_masks if prompts is not None else None, image_size, channels, n_T, cfg_scale, device
        )

        if 'attn_save_dir' in entry:
            perform_attention_visualization(attn_ctrl, entry, token_labels_for_first, prompts, classes)

        os.makedirs(save_path, exist_ok=True)
        x_vis = x_i * 0.5 + 0.5
        x_vis = x_vis.clamp(0, 1)
        if prompts is not None:
            B = len(prompts)
        else:
            B = len(classes)
        nrow = min(4, max(1, B))
        grid = make_grid(x_vis.float(), nrow=nrow)
        img = grid.permute(1, 2, 0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_path, "sample_qformer_infer_last.png"))

if __name__ == "__main__":
    default_device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="DDPM sampling with DiT + QFormer text conditioning (CIFAR10)")
    parser.add_argument("--config_yaml", type=str, required=True, help="YAML file containing Image_Generation_Configs list of dictionaries")
    parser.add_argument("--device", type=str, default=default_device, help="Device to run inference on")

    args = parser.parse_args()
    
    eval_cifar10_qformer(args)
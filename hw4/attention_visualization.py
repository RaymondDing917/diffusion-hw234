import os
import argparse
import json
import random
from typing import Dict, List, Tuple, Optional

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
from dit import DiT_Llama
import math
import torch
from dit import Attention as _DiTAttention
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile

def enable_qformer_cross_attention_patch(model: DiT_Llama, capture_store: List[Dict] = None):
    """
    Patch QueryEmbedder cross-attention blocks. Starter behavior: identical output.
    Solution: when capture_store is provided, record attention maps per forward with
    metadata: {layer, attn: Tensor[B,H,Q,K], timestep}.
    """

    if not hasattr(model, "query_embedder") or model.query_embedder is None:
        raise ValueError("Model has no QueryEmbedder. Set use_text_conditioning=True when building the model.")


    def _make_wrapper(layer_idx: int, attn_module):
        orig_forward = attn_module.forward

        def wrapped(x, freqs_cis=None, cross_attention_states=None, cross_attention_mask=None):
            bsz, seqlen_q, _ = x.shape
            n_heads = attn_module.n_heads
            head_dim = attn_module.head_dim

            xq = attn_module.wq(x)
            if cross_attention_states is not None:
                xk_in = cross_attention_states
                xv_in = cross_attention_states
            else:
                xk_in = x
                xv_in = x
            xk = attn_module.wk(xk_in)
            xv = attn_module.wv(xv_in)

            dtype = xq.dtype
            xq = attn_module.q_norm(xq)
            xk = attn_module.k_norm(xk)

            seqlen_kv = xk_in.shape[1]
            xq = xq.view(bsz, seqlen_q, n_heads, head_dim)
            xk = xk.view(bsz, seqlen_kv, n_heads, head_dim)
            xv = xv.view(bsz, seqlen_kv, n_heads, head_dim)

            if attn_module.use_rotary:
                xq, xk = _DiTAttention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            xq, xk = xq.to(dtype), xk.to(dtype)

            q = xq.permute(0, 2, 1, 3)
            k = xk.permute(0, 2, 1, 3)
            v = xv.permute(0, 2, 1, 3)

            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, Q, K)
            if cross_attention_states is not None and cross_attention_mask is not None:
                # cross_attention_mask: (B, K) where True indicates valid tokens
                mask_bool = cross_attention_mask.to(dtype=torch.bool)
                attn_mask = (~mask_bool).unsqueeze(1).unsqueeze(2)  # (B,1,1,K) True=mask
                scores = scores.masked_fill(attn_mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)  # (B, H, Q, D)

            # Record attention map in capture_store, along with metadata about the layer and timestep. 
            #       Make sure to put the saved tensor in CPU memory.
            if capture_store is not None:
                timestep = None
                if hasattr(model, "_attn_capture_ctx") and isinstance(model._attn_capture_ctx, dict):  # type: ignore[attr-defined]
                    timestep = int(model._attn_capture_ctx.get("timestep", -1))  # type: ignore[attr-defined]
                capture_store.append({"layer": layer_idx, "attn": attn.detach().cpu(), "timestep": timestep})
            
            out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seqlen_q, n_heads * head_dim)
            return attn_module.wo(out)

        attn_module.forward = wrapped

    for layer_idx, blk in enumerate(model.query_embedder.blocks):
        _make_wrapper(layer_idx, blk["cross_attn"])  # type: ignore[index]

    if not hasattr(model, "_attn_capture_ctx"):
        model._attn_capture_ctx = {"timestep": None}  # type: ignore[attr-defined]
    
    return {"captured": capture_store}


def view_qformer_attn(attn_files, out_paths, titles, batch_idx=0, dpi=150):
    if len(attn_files) != len(out_paths):
        raise ValueError("attn_files and out_paths must have the same length")
    if titles is not None and len(titles) != len(attn_files):
        raise ValueError("titles must be the same length as attn_files when provided")

    def _load_payload(path):
        if path.endswith(".pt"):
            return torch.load(path, map_location="cpu")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _extract_attn_list(payload):
        # Normalize to a list of tensors with shape [B, H, Q, K]
        if isinstance(payload, dict) and "attn" in payload:
            entries = payload["attn"]
        else:
            entries = payload
        # entries can be a list of dicts, a single tensor, or a list of tensors
        out_tensors = []
        if isinstance(entries, (list, tuple)):
            for e in entries:
                if isinstance(e, dict):
                    t = e.get("attn", None)
                    if t is None:
                        continue
                    if isinstance(t, np.ndarray):
                        t = torch.from_numpy(t)
                    out_tensors.append(t)
                elif torch.is_tensor(e):
                    out_tensors.append(e)
                elif isinstance(e, np.ndarray):
                    out_tensors.append(torch.from_numpy(e))
        elif torch.is_tensor(entries):
            out_tensors.append(entries)
        elif isinstance(entries, np.ndarray):
            out_tensors.append(torch.from_numpy(entries))
        return out_tensors

    for i, fpath in enumerate(attn_files):
        payload = _load_payload(fpath)
        tensors = _extract_attn_list(payload)
        if len(tensors) == 0:
            continue
        # Stack over layers/records and average over layers and heads
        # shapes assumed [B, H, Q, K]
        try:
            stacked = torch.stack([t.float() for t in tensors], dim=0)  # [L, B, H, Q, K]
        except Exception:
            # If shapes mismatch, skip this file
            continue
        A = stacked.mean(dim=(0, 2))  # [B, Q, K]
        b = max(0, min(batch_idx, A.size(0) - 1))
        A_b = A[b].numpy()  # [Q, K]

        # Render heatmap
        Q, K = A_b.shape
        fig = plt.figure(figsize=(max(6, 0.5 * K), max(4, 0.5 * Q)), dpi=dpi)
        ax = fig.add_subplot(111)
        im = ax.imshow(A_b, aspect='auto', interpolation='nearest', cmap='viridis')
        # Prepare x-axis token labels if available in the payload; otherwise fall back to indices
        token_labels = None
        if isinstance(payload, dict):
            raw_tokens = payload.get("tokens", None)
            if raw_tokens is not None:
                if isinstance(raw_tokens, (list, tuple)):
                    token_labels = [str(t) for t in raw_tokens]
                elif isinstance(raw_tokens, np.ndarray):
                    token_labels = [str(t) for t in raw_tokens.tolist()]
        if token_labels is None:
            token_labels = [str(j) for j in range(K)]
        if len(token_labels) < K:
            token_labels = token_labels + [str(j) for j in range(len(token_labels), K)]
        elif len(token_labels) > K:
            token_labels = token_labels[:K]
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('tokens (K)')
        ax.set_ylabel('query index (Q)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        title = None
        if titles is not None:
            title = titles[i]
        if title is None:
            title = os.path.splitext(os.path.basename(fpath))[0]
        ax.set_title(title)

        out_path = out_paths[i]
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

def setup_attention_capture(
    entry: Dict,
    model: DiT_Llama,
    token_labels_for_first: Optional[List[str]]
) -> Optional[Dict]:
    """
    Setup attention capture for QFormer cross-attention visualization.
    
    Args:
        entry: Configuration dictionary containing attn_save_dir and attn_timesteps
        model: The DiT model with QueryEmbedder
        token_labels_for_first: Optional token labels for visualization
        
    Returns:
        Dictionary with captured attention data or None if not configured
    """
    attn_save_dir = entry.get("attn_save_dir", None)
    attn_timesteps = entry.get("attn_timesteps", None)
    
    if attn_save_dir is None or attn_timesteps is None or len(attn_timesteps) == 0:
        return None
    
    os.makedirs(attn_save_dir, exist_ok=True)
    captured: List[Dict] = []
    attn_ctrl = enable_qformer_cross_attention_patch(model, capture_store=captured)
    
    # Provide token labels to capture context so each record includes them
    try:
        if token_labels_for_first is not None and hasattr(model, "_attn_capture_ctx") and isinstance(model._attn_capture_ctx, dict):  # type: ignore[attr-defined]
            model._attn_capture_ctx["tokens"] = list(token_labels_for_first)
    except Exception:
        pass
    
    return attn_ctrl

def perform_attention_visualization(
    attn_ctrl: Dict,
    entry: Dict,
    token_labels_for_first: Optional[List[str]],
    prompts: Optional[List[str]],
    classes: Optional[List[int]]
) -> None:
    """
    Visualize captured attention maps and save to files.
    
    Args:
        attn_ctrl: Dictionary containing captured attention data
        entry: Configuration dictionary containing attn_save_dir and attn_timesteps
        token_labels_for_first: Optional token labels for the first prompt
        prompts: Optional list of prompts used for generation
        classes: Optional list of classes used for generation
    """
    if attn_ctrl is None:
        return
    
    attn_save_dir = entry.get("attn_save_dir")
    attn_timesteps = entry.get("attn_timesteps")
    
    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    attn_files: List[str] = []
    out_paths: List[str] = []
    titles: List[str] = []
    
    for t in attn_timesteps:
        attn_for_t = [rec for rec in attn_ctrl["captured"] if int(rec.get("timestep", -1)) == int(t)]
        tokens_for_payload = token_labels_for_first
        # replace the BPE space marker for prettier plots
        if tokens_for_payload is not None:
            tokens_for_payload = [t.replace('Ġ', ' ').replace('Ċ', '\n') for t in tokens_for_payload]
        payload = {
            "timestep": int(t),
            "prompts": list(prompts) if prompts is not None else None,
            "classes": list(classes) if classes is not None else None,
            "tokens": list(tokens_for_payload) if tokens_for_payload is not None else None,
            "attn": attn_for_t,
        }
        tmp_path = os.path.join(tmp_dir, f"qformer_attn_t{int(t)}.pt")
        torch.save(payload, tmp_path)
        attn_files.append(tmp_path)
        out_paths.append(os.path.join(attn_save_dir, f"qformer_attn_t{int(t)}.png"))
        titles.append(f"t={int(t)}")

    view_qformer_attn(attn_files, out_paths, titles)
    

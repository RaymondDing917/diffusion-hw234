import os
import argparse
import json
import random
from typing import Dict, List, Tuple, Optional
import re
import torch
import torch.nn as nn
import gc
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb

def get_cifar_names():
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

def get_synonyms():
    return {
        "airplane": ["vehicle that flies"],
        "automobile": ["small vehicle that drives"],
        "bird": ["animal that flies"],
        "cat": ["four legged animal that purrs"],
        "deer": ["four legged animal with antlers"],
        "dog": ["four legged animal that barks"],
        "frog": ["four legged animal that hops"],
        "horse": ["four legged animal with a mane and sometimes a saddle"],
        "ship": ["vehicle that floats"],
        "truck": ["big vehicle that drives"],
    }

def get_text_captions(path: str):
    cifar10_names = get_cifar_names()
    num_classes = len(get_cifar_names())
    index_to_captions: Dict[int, List[str]] = {}
    captions_per_class: Dict[int, List[str]] = {i: [] for i in range(num_classes)}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dense captions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            item = json.loads(line)
            idx_val = int(item["index"])  # original CIFAR-10 train index
            index_to_captions.setdefault(idx_val, []).append(item["caption"])
            if "label_index" in item:
                lbl = int(item["label_index"])
                if 0 <= lbl < num_classes:
                    captions_per_class[lbl].append(item["caption"])

    for idx in range(num_classes):
        if len(captions_per_class[idx]) == 0:
            captions_per_class[idx].append(f"a photo of a {cifar10_names[idx]}")
    return index_to_captions, captions_per_class


class PromptVariantDataset(torch.utils.data.Dataset):
    """
    Stage 1 dataset: select one prompt variant per sample index.
    """

    def __init__(
        self,
        cifar10_dataset: datasets.CIFAR10,
        cifar10_names: List[str],
        synonyms: Dict[str, List[str]],
        index_to_captions: Optional[Dict[int, List[str]]] = None,
        caption_use_prob: float = 0.8,
        synonym_replace_prob: float = 0.5,
    ):
        self.cifar10_dataset = cifar10_dataset
        self.cifar10_names = cifar10_names
        self.synonyms = synonyms
        self.index_to_captions = index_to_captions or {}
        self.caption_use_prob = float(caption_use_prob)
        self.synonym_replace_prob = float(synonym_replace_prob)
        self._selected_keys: List[str] = []
        self._selected_texts: List[str] = []
        self._select_all_text_prompts()

    def _select_all_text_prompts(self) -> None:
        targets = getattr(self.cifar10_dataset, "targets", None)
        for idx in range(len(self.cifar10_dataset)):
            if targets is not None:
                label_int = int(targets[idx])
            else:
                _, label = self.cifar10_dataset[idx]
                label_int = int(label)

            base = f"a photo of a {self.cifar10_names[label_int]}"
            caps = self.index_to_captions.get(int(idx), [])
            key = self._select_prompt_variant_key(has_caption=(len(caps) > 0))
            if key.startswith("caption") and len(caps) > 0:
                text = random.choice(caps)
            else:
                text = base
            if key.endswith("_syn"):
                text = self._maybe_apply_synonym(text, label_int, self.cifar10_names, self.synonyms, 1.0)

            self._selected_keys.append(key)
            self._selected_texts.append(text)
    
    def _select_prompt_variant_key(self, has_caption: bool) -> str:
        use_caption = bool(has_caption) and (random.random() < self.caption_use_prob)
        use_synonym = random.random() < self.synonym_replace_prob
        if use_caption:
            return "caption_syn" if use_synonym else "caption"
        return "base_syn" if use_synonym else "base"

    def _maybe_apply_synonym(self, text: str, class_index: int, cifar10_names: List[str], synonyms: Dict[str, List[str]], prob: float) -> str:
        class_name = cifar10_names[class_index]
        if random.random() >= prob:
            return text
        candidates = synonyms.get(class_name, [])
        if not candidates:
            return text
        synonym = random.choice(candidates)

        pattern = re.compile(rf"\b{re.escape(class_name)}\b", flags=re.IGNORECASE)
        return pattern.sub(synonym, text)

    def __len__(self) -> int:
        return len(self.cifar10_dataset)

    def __getitem__(self, idx: int):
        img, label = self.cifar10_dataset[idx]
        label_int = int(label)
        key = self._selected_keys[int(idx)]
        text = self._selected_texts[int(idx)]
        # Tuple contains: image tensor, label int, CIFAR10 index, variant key, text
        return img, label_int, int(idx), key, text


class HiddenStateDataset(torch.utils.data.Dataset):
    """
    Stage 2 dataset: replace selected prompt text with cached hidden states.
    """

    def __init__(
        self,
        prompt_dataset: PromptVariantDataset,
        device: torch.device,
        tokenizer=None,
        gpt2=None,
        gpt2_layer_index: Optional[int] = None,
        cache_build_batch_size: int = 512,
    ):
        self.prompt_dataset = prompt_dataset
        self.device = device
        self.cache = HiddenStateCache(
            prompt_list=self.prompt_dataset._selected_texts,
            tokenizer=tokenizer,
            gpt2=gpt2,
            gpt2_layer_index=gpt2_layer_index,
            device=self.device,
            cache_build_batch_size=cache_build_batch_size,
        )

    def __len__(self) -> int:
        return len(self.prompt_dataset)

    def __getitem__(self, idx: int):
        idx_val = int(idx)
        img, label, _, _, text = self.prompt_dataset[idx_val]
        states, masks = self.cache.get(text)
        return img, int(label), idx_val, text, states, masks

    def _pad_states_and_masks(self,
        per_prompt_states: List[torch.Tensor],
        per_prompt_masks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        max_len = max(hs.size(0) for hs in per_prompt_states) if per_prompt_states else 0
        padded_states: List[torch.Tensor] = []
        padded_masks: List[torch.Tensor] = []
        for hs, mask in zip(per_prompt_states, per_prompt_masks):
            pad_len = max_len - hs.size(0)
            if pad_len > 0:
                pad = torch.zeros(pad_len, hs.size(1), device=hs.device, dtype=hs.dtype)
                hs = torch.cat([hs, pad], dim=0)
                mask_pad = torch.zeros(pad_len, dtype=torch.bool, device=mask.device)
                mask = torch.cat([mask.to(torch.bool), mask_pad], dim=0)
            padded_states.append(hs)
            padded_masks.append(mask.to(torch.bool))
        return torch.stack(padded_states, dim=0).to(device), torch.stack(padded_masks, dim=0).to(device)

    def collate_fn(self, batch: List[Tuple[torch.Tensor, int, int, str, torch.Tensor, torch.Tensor]]):
        imgs, labels, _, _, states_list, masks_list = zip(*batch)
        imgs_t = torch.stack(list(imgs), dim=0)
        labels_t = torch.tensor(list(labels), dtype=torch.long)
        states, masks = self._pad_states_and_masks(list(states_list), list(masks_list))
        return (imgs_t, states, masks), labels_t

    def build_one_per_class_batch(self, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Build one pseudo-validation batch by sampling one example per class from this dataset."""
        label_to_indices: Dict[int, List[int]] = {lbl: [] for lbl in range(num_classes)}
        base_targets = getattr(self.prompt_dataset.cifar10_dataset, "targets", None)

        for idx in range(len(self)):
            if base_targets is not None:
                lbl = int(base_targets[idx])
            else:
                _, lbl, _, _, _ = self.prompt_dataset[idx]
                lbl = int(lbl)
            if 0 <= lbl < num_classes:
                label_to_indices[lbl].append(idx)

        chosen_indices: List[int] = []
        for lbl in range(num_classes):
            candidates = label_to_indices.get(lbl, [])
            if len(candidates) == 0:
                raise ValueError(f"No samples available for class {lbl}; cannot build validation batch.")
            chosen_indices.append(random.choice(candidates))

        batch_items = [self[idx] for idx in chosen_indices]
        prompts = [item[3] for item in batch_items]
        xy, labels = self.collate_fn(batch_items)
        _, text_states, text_masks = xy
        return labels, text_states, text_masks, prompts


class HiddenStateCache:
    """
    In-memory map: prompt text -> {"states": Tensor, "masks": Tensor}.
    """

    def __init__(
        self,
        prompt_list: List[str],
        *,
        tokenizer=None,
        gpt2=None,
        gpt2_layer_index: Optional[int] = None,
        device: Optional[torch.device] = None,
        cache_build_batch_size: int = 512,
    ):
        self.prompts = list(dict.fromkeys(prompt_list))
        self.tokenizer = tokenizer
        self.gpt2 = gpt2
        self.gpt2_layer_index = gpt2_layer_index
        self.device = device
        self.cache_build_batch_size = int(cache_build_batch_size)
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.tokenizer is None or self.gpt2 is None or self.gpt2_layer_index is None or self.device is None:
            raise ValueError(
                "Provide tokenizer/gpt2/gpt2_layer_index/device so HiddenStateCache can build in-memory states."
            )
        self._cache = self._build_cache()

    def _build_cache(self) -> Dict[str, Dict[str, torch.Tensor]]:
        agg: Dict[str, Dict[str, torch.Tensor]] = {}
        N = len(self.prompts)

        self.gpt2.eval()
        with torch.no_grad():
            for start in tqdm(range(0, N, self.cache_build_batch_size), desc="Caching text embeddings"):
                end = min(N, start + self.cache_build_batch_size)
                prompts = self.prompts[start:end]

                states, masks = prompts_to_padded_hidden_states(
                    prompts=prompts,
                    gpt2=self.gpt2,
                    tokenizer=self.tokenizer,
                    gpt2_layer_index=int(self.gpt2_layer_index),
                    device=self.device,
                )

                for j, text in enumerate(prompts):
                    agg[text] = {
                        "states": states[j].detach().cpu(),
                        "masks": masks[j].detach().cpu(),
                    }
        return agg

    def get(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self._cache.get(text)
        if rec is None:
            raise KeyError(f"Cache entry for prompt text missing: {text[:80]}")
        if "states" not in rec or "masks" not in rec:
            raise KeyError(f"Cache entry for prompt text is missing 'states' or 'masks': {text[:80]}")
        states = rec["states"]
        masks = rec["masks"]
        if not torch.is_tensor(states):
            states = torch.as_tensor(states)
        if not torch.is_tensor(masks):
            masks = torch.as_tensor(masks, dtype=torch.bool)
        else:
            masks = masks.to(torch.bool)
        return states, masks


def prompts_to_padded_hidden_states(
    prompts: List[str],
    gpt2,
    tokenizer,
    gpt2_layer_index: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of text prompts to a single padded tensor of GPT-2 hidden states.

    Returns a tuple (hidden_states, masks) where
    hidden_states: (B, max_seq_len, C) are the padded hidden states for each prompt from the layer specified by gpt2_layer_index
    masks: (B, max_seq_len) is a boolean mask containing True for valid tokens and False for padding (dtype bool)
    
    B is the batch size, max_seq_len is the length of the longest prompt, and C is the hidden dimension
    """
    if not prompts:
        raise ValueError("prompts must contain at least one text prompt")
    if gpt2_layer_index < 0:
        raise ValueError("gpt2_layer_index must be non-negative")

    # Hugging Face tokenizers support batch padding, which keeps cache creation fast.
    if hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token"):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gpt2(**encoded, output_hidden_states=True)
        if gpt2_layer_index >= len(outputs.hidden_states):
            raise ValueError("gpt2_layer_index is outside the available hidden states")
        masks = encoded["attention_mask"].to(torch.bool)
        states = outputs.hidden_states[gpt2_layer_index]
        states = states.masked_fill(~masks.unsqueeze(-1), 0)
        return states, masks

    # Simple tokenizers may only accept one prompt and provide no padding token.
    per_prompt_states = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = gpt2(**encoded, output_hidden_states=True)
        if gpt2_layer_index >= len(outputs.hidden_states):
            raise ValueError("gpt2_layer_index is outside the available hidden states")
        per_prompt_states.append(outputs.hidden_states[gpt2_layer_index][0].to(device))

    max_len = max(states.size(0) for states in per_prompt_states)
    padded_states = []
    masks = []
    for states in per_prompt_states:
        length = states.size(0)
        padded_states.append(torch.cat([
            states,
            states.new_zeros(max_len - length, states.size(-1)),
        ], dim=0))
        masks.append(torch.arange(max_len, device=states.device) < length)
    return torch.stack(padded_states), torch.stack(masks)

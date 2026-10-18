import os
import argparse
import json
import random
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import torch
import torch.nn as nn
import gc
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from dit import DiT_Llama
from ddpm import ddpm_schedules
from image_caption_data import (
    get_text_captions,
    PromptVariantDataset,
    HiddenStateDataset,
)
from image_caption_data import get_cifar_names, get_synonyms

def setup_optimizer_and_scheduler(model: DiT_Llama, args: argparse.Namespace):
    """
    Setup the optimizer and scheduler for the model given the arguments in args.
    
    The parameters of QFormer should be trainable, and all of the DiT parameters should be frozen.
    Set up the AdamW optimizer with the given arguments, as well as a scheduler with linear warmup if the warmup_steps argument is set.
    set the criterion to be MSE loss.
    Return the optimizer, scheduler, and criterion.
    """
    
    # TODO: Set the correct parameters to be trainable, set up the optimizer and scheduler, and the criterion
    # ====== BEGIN STUDENT SOLUTION ===========================================
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("query_embedder.")
    optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
    weight_decay=args.weight_decay,
    eps=args.adam_epsilon,
    betas=(0.9, 0.95),
)
    if args.warmup_steps > 0:
        def lr_lambda(step):
            return min(1.0, (step + 1) / args.warmup_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    else:
        scheduler = None
    criterion = nn.MSELoss()


    # ====== END STUDENT SOLUTION ===========================================
    return optimizer, scheduler, criterion

def maybe_load_resume_states(model: DiT_Llama, args: argparse.Namespace, device: torch.device, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, steps_per_epoch: int):
    # If provided, load model/optimizer states for resuming
    if args.resume_model_path is not None and os.path.isfile(args.resume_model_path):
        resume_state = torch.load(args.resume_model_path, map_location=device)
        model.load_state_dict(resume_state, strict=True)
    if args.resume_optimizer_path is not None and os.path.isfile(args.resume_optimizer_path):
        opt_state = torch.load(args.resume_optimizer_path, map_location=device)
        optimizer.load_state_dict(opt_state)
    if scheduler is not None and args.start_epoch > 0:
            for _ in range(args.start_epoch * steps_per_epoch):
                scheduler.step()
    return model, optimizer, scheduler

def run_qformer_inference(
    model: DiT_Llama,
    sched: Dict[str, torch.Tensor],
    classes: Optional[torch.Tensor],
    text_states: Optional[torch.Tensor],
    text_masks: Optional[torch.Tensor],
    image_size: int,
    channels: int,
    n_T: int,
    cfg_scale: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if classes is not None:
            B = classes.size(0)
        else:
            B = text_states.size(0)
        x_i = torch.randn(B, channels, image_size, image_size, device=device)
        for i in range(n_T, 0, -1):
            if hasattr(model, "_attn_capture_ctx") and isinstance(model._attn_capture_ctx, dict):  # type: ignore[attr-defined]
                model._attn_capture_ctx["timestep"] = int(i)  # type: ignore[attr-defined]
            t_scalar = torch.full((B,), float(i) / n_T, device=x_i.device)
            if cfg_scale > 0.0:
                # Conditional path: use QFormer text states
                eps_cond = model(x_i, t_scalar, classes, text_hidden_states=text_states, text_attention_mask=text_masks)
                # Unconditional path: use unconditional embedding from pretrained DiT model (before qformer training)
                cfg_index = getattr(model.y_embedder, "num_classes", 10)
                y_uncond = torch.full((B,), int(cfg_index), device=device, dtype=torch.long)
                eps_uncond = model(
                    x_i,
                    t_scalar,
                    y_uncond,
                    text_hidden_states=None,
                )
                eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_hat = model(x_i, t_scalar, classes, text_hidden_states=text_states, text_attention_mask=text_masks)

            oneover_sqrta = sched["oneover_sqrta"][i]
            mab_over_sqrtmab = sched["mab_over_sqrtmab"][i]
            sqrt_beta_t = sched["sqrt_beta_t"][i]
            z = torch.randn_like(x_i) if i > 1 else 0
            x_i = oneover_sqrta * (x_i - eps_hat * mab_over_sqrtmab) + sqrt_beta_t * z
        return x_i

def train_cifar10_qformer_dense_captions(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    wandb.init(project=f"qformer_training", name=f"initial_path_{os.path.basename(args.pretrained_model_path).split('.')[0]}_{args.epochs}_epochs_{args.batch_size}_batch_size")

    channels = 3
    num_classes = 10
    image_size = 32

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # index_to_captions maps an index to dense captions for that exact training example.
    index_to_captions, _ = get_text_captions(args.dense_captions_path)

    os.makedirs(args.data_dir, exist_ok=True)
    cifar10_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    if args.max_train_examples is not None:
        num_examples = min(args.max_train_examples, len(cifar10_dataset))
        cifar10_dataset = Subset(cifar10_dataset, list(range(num_examples)))
    cifar10_names = get_cifar_names()
    synonyms = get_synonyms()

    prompt_dataset = PromptVariantDataset(
        cifar10_dataset=cifar10_dataset,
        cifar10_names=cifar10_names,
        synonyms=synonyms,
        index_to_captions=index_to_captions,
        caption_use_prob=args.caption_use_prob,
        synonym_replace_prob=args.synonym_replace_prob,
    )
    model = DiT_Llama(
        in_channels=channels,
        input_size=image_size,
        dim=256,
        n_layers=10,
        n_heads=8,
        num_classes=num_classes,
        use_text_conditioning=True, # this turns on the qformer
        transformer_hidden_size=args.transformer_hidden_size,
        num_query_tokens=args.num_query_tokens,
    ).to(device)

    state = torch.load(args.pretrained_model_path, map_location=device)
    model.load_state_dict(state, strict=False)

    optimizer, scheduler, criterion = setup_optimizer_and_scheduler(model, args)

    sched = ddpm_schedules(args.beta1, args.beta2, args.n_T)
    for k in list(sched.keys()):
        sched[k] = sched[k].to(device)

    os.makedirs(args.gpt2_cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=args.gpt2_cache_dir)
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=args.gpt2_cache_dir).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = HiddenStateDataset(
        prompt_dataset=prompt_dataset,
        device=device,
        tokenizer=tokenizer,
        gpt2=gpt2,
        gpt2_layer_index=args.gpt2_layer_index,
        cache_build_batch_size=max(64, args.batch_size),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )

    # Pick a fixed batch of one example per class for evaluation and visualization at the end of each epoch
    val_labels, val_text_states, val_text_masks, val_prompts = dataset.build_one_per_class_batch(
        num_classes=num_classes,
    )

    full_steps_per_epoch = len(dataloader)
    if args.max_batches_per_epoch is None:
        steps_per_epoch = full_steps_per_epoch
    else:
        steps_per_epoch = min(full_steps_per_epoch, args.max_batches_per_epoch)

    # load from optimizer and model checkpoints if provided, i.e. if your training was interrupted, you can resume from the checkpoint
    model, optimizer, scheduler = maybe_load_resume_states(model, args, device, optimizer, scheduler, steps_per_epoch)


    # Setup resume and checkpointing utilities
    start_epoch = args.start_epoch
    global_step = start_epoch * steps_per_epoch
    ckpt_dir = args.optimizer_ckpt_dir if args.optimizer_ckpt_dir else os.path.join(args.save_dir, "optimizer_ckpts")

    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    total_steps = args.epochs * steps_per_epoch
    pbar = tqdm(total=total_steps, desc="Training")
    for epoch in range(args.epochs):
        # Skip full epochs if resuming beyond them
        if epoch < start_epoch:
            pbar.update(steps_per_epoch)
            continue
        loss_ema = None
        for step_idx, (xy, y) in enumerate(dataloader):
            if step_idx >= steps_per_epoch:
                break
            x_img, text_states, text_masks = xy
            x = x_img.to(device)
            y = y.to(device)
            text_masks = text_masks.to(device)

            b = x.size(0)
            optimizer.zero_grad()

            t_int = torch.randint(1, args.n_T + 1, (b,), device=x.device)
            t = t_int.float() / args.n_T
            eps = torch.randn_like(x)

            # add noise to the image
            x_t = sched["sqrtab"][t_int, None, None, None] * x + sched["sqrtmab"][t_int, None, None, None] * eps

            # model forward pass
            eps_pred = model(x_t, t, y, text_hidden_states=text_states, text_attention_mask=text_masks)

            loss = criterion(eps_pred, eps)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

            loss_item = loss.item()
            wandb.log({"loss": loss_item})

            if loss_ema is None:
                loss_ema = loss_item
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss_item
                wandb.log({"loss_ema": loss_ema})
            global_step += 1
            pbar.set_description(f"Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps} | loss: {loss_ema:.4f}")
            pbar.update(1)

        model.eval()
        with torch.no_grad():
            x_i = run_qformer_inference(
                model,
                sched,
                val_labels.to(device),
                val_text_states.to(device),
                val_text_masks.to(device),
                image_size,
                channels,
                args.n_T,
                args.cfg,
                device,
            )
            x_vis = x_i * 0.5 + 0.5
            x_vis = x_vis.clamp(0, 1)
            grid = make_grid(x_vis.float(), nrow=4)
            img = grid.permute(1, 2, 0).detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(args.save_dir, f"sample_qformer_dense_{epoch}_last.png"))

            try:
                wandb.log({
                    "samples/grid": wandb.Image(grid, caption="epoch %d dense grid" % epoch)
                })
                per_sample_images = []
                for idx in range(x_vis.size(0)):
                    single = x_vis[idx]
                    caption = val_prompts[idx] if idx < len(val_prompts) else ""
                    per_sample_images.append(wandb.Image(single, caption=caption))
                wandb.log({"samples/per_sample": per_sample_images})
            except Exception:
                pass
            os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_model_path)

            # Save optimizer (and paired model weights) every N epochs
            if (epoch + 1) % int(args.optimizer_ckpt_interval) == 0:
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    opt_path = os.path.join(ckpt_dir, f"optimizer_epoch_{epoch+1}.pt")
                    mdl_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
                    torch.save(optimizer.state_dict(), opt_path)
                    torch.save(model.state_dict(), mdl_path)
                except Exception:
                    pass

            try:
                del x_i
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                gc.collect()
            except Exception:
                pass
        model.train()

def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    default_device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="DDPM training with DiT + QFormer using dense captions (CIFAR10)")
    parser.add_argument("--pretrained_model_path", type=str, default="./data/ddpm_dit_cifar_100_epochs.pth", help="Path to the pretrained DiT model")
    parser.add_argument("--dense_captions_path", type=str, default="./data/cifar10_dense_captions.jsonl", help="Path to CIFAR-10 dense captions jsonl")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10", help="Directory to save the data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for the AdamW optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the training")
    parser.add_argument("--n_T", type=int, default=1000, help="Number of timesteps for the DDPM scheduler")
    parser.add_argument("--beta1", type=float, default=1e-4, help="Beta1 for the DDPM scheduler")
    parser.add_argument("--beta2", type=float, default=2e-2, help="Beta2 for the DDPM scheduler")
    parser.add_argument("--device", type=str, default=default_device, help="Device to train on")
    parser.add_argument("--save_dir", type=str, default="./contents_qformer_dense_training", help="Directory to save the model and optimizer checkpoints")
    parser.add_argument("--save_model_path", type=str, default="./models/ddpm_dit_cifar_qformer_dense.pth", help="Path to save the model")
    parser.add_argument("--num_query_tokens", type=int, default=4, help="Number of query tokens for the QFormer")
    parser.add_argument("--transformer_hidden_size", type=int, default=768, help="hidden size of GPT-2 base")
    parser.add_argument("--cfg", type=float, default=0.0, help="classifier-free guidance scale at epoch-end sampling")
    parser.add_argument("--warmup_steps", type=int, default=50, help="linear LR warmup steps for query embedder optimizer")
    parser.add_argument("--gpt2_layer_index", type=int, default=12, help="0-based GPT-2 hidden-state index to use for conditioning (default: 12, the last GPT-2 base state)")
    parser.add_argument("--gpt2_cache_dir", type=str, default="./data/gpt2", help="HuggingFace cache directory for GPT-2")
    parser.add_argument("--synonym_replace_prob", type=float, default=0.5, help="Probability to replace the class name with a synonym in the caption")
    parser.add_argument("--caption_use_prob", type=float, default=0.8, help="Probability to use the dense caption; else base prompt")
    parser.add_argument("--optimizer_ckpt_interval", type=int, default=5, help="Save optimizer state every N epochs (and paired model weights)")
    parser.add_argument("--optimizer_ckpt_dir", type=str, default=None, help="Directory to save optimizer state checkpoints; defaults to save_dir/optimizer_ckpts")
    parser.add_argument("--resume_optimizer_path", type=str, default=None, help="Path to an optimizer state dict to resume from")
    parser.add_argument("--resume_model_path", type=str, default=None, help="Path to model weights matching the optimizer state")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from when resuming")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None, help="Maximum number of training batches to process per epoch (default: all)")
    parser.add_argument("--max_train_examples", type=int, default=None, help="If set, keep only the first N training examples in PromptVariantDataset")
    parser.add_argument("--seed", type=int, default=10623, help="Random seed for reproducibility")
    args = parser.parse_args()
    if args.max_batches_per_epoch is not None and args.max_batches_per_epoch <= 0:
        parser.error("--max_batches_per_epoch must be a positive integer")
    if args.max_train_examples is not None and args.max_train_examples <= 0:
        parser.error("--max_train_examples must be a positive integer")
    
    set_seed(args.seed)

    train_cifar10_qformer_dense_captions(args)

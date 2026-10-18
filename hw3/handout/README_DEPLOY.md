# HW3 cloud run

The files in `hw3_upload.zip` are sufficient to upload to a Linux GPU instance. The archive omits `wandb_api.json`, caches, and local results.

Verified instance: one full RTX 5090 (32 GB) with Paratera's PyTorch 2.7.0 image and `/root/shared-nvme` mounted. The runner writes all checkpoints, predictions, logs, and the Hugging Face cache to that persistent mount.

```bash
unzip hw3_upload.zip -d ~/hw3
cd ~/hw3
env -u PIP_CONSTRAINT python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python test_lora.py
bash run_hw3.sh default
```

The Paratera PyTorch 2.7.0 image sets a global `PIP_CONSTRAINT` that pins `dill==0.3.9`, which conflicts with `datasets==3.6.0`; `env -u PIP_CONSTRAINT` applies only to this install command. If Hugging Face downloads time out, use Paratera's [academic network acceleration instructions](https://ai.paratera.com/document/container/faq/env/network_turbo) in the same shell before running the script. Do not copy the proxy credentials into this archive.

`default` trains GPT-2 Medium with LoRA rank 128 and alpha 512 for 80 iterations. After completion, `r128_a512/results.json` contains **best validation checkpoint** test accuracy; `validation.jsonl` contains the validation loss curve and `train.log` contains the full console output. Paths are rooted at `/root/shared-nvme/hw3` by default. Set `RUN_ROOT` to another persistent directory if necessary.

Other modes are `baseline`, `r16_a64`, `r16_a256`, `r196_a784`, `full`, and `prompt_short`. `all` runs every mode sequentially. The short prompt is the alternative wording for the prompt experiment; the original prompt remains the default. A completed mode is skipped when rerunning the script, and a mode with an existing best checkpoint resumes training. The runner uses offline WandB logging unless `WANDB_MODE` is set to `online` and `wandb_api.json` contains a key.

The training script evaluates the best checkpoint automatically. Add `--evaluate_last` to a manual `train.py` command only when last-checkpoint accuracy is needed. The runner saves `ckpt_last.pt` even when that extra test evaluation is skipped.

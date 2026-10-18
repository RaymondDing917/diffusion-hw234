"""Run cat training locally with bounded CPU use and offline logging."""
import os
from pathlib import Path
import runpy

local_tmp = Path(__file__).resolve().parent / "results" / "tmp"
local_tmp.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(local_tmp)
os.environ["TMP"] = str(local_tmp)
os.environ["WANDB_MODE"] = "disabled"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["WANDB_DIR"] = str(Path(__file__).resolve().parent / "results")
os.chdir(Path(__file__).resolve().parent)

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
print("Local training: CPU threads=4; local logs; W&B disabled", flush=True)
runpy.run_path("main.py", run_name="__main__")

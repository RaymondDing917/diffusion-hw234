#!/usr/bin/env bash
# These custom files were created for 10-x23 Homework 4 and are hosted in Google Drive. 
set -euo pipefail; mkdir -p data
gdown -O data/cifar10_dense_captions.jsonl "https://drive.google.com/uc?id=1vkVzNXcRiyNIqrzUurb6huvfPfCVRiti"
gdown -O data/ddpm_dit_cifar_100_epochs.pth "https://drive.google.com/uc?id=1B9uDE1ISAKu2gu181tJ8P5TD0zZ2j1T6"

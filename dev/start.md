# Example

https://veomni.readthedocs.io/en/latest/usage/basic_modules.html#usage

## Setup

First setup `.env`

- make sure the uv cache are in the same file system. Otherwise the python will not be accessible across shared file system, as it is linked in one of the node’s unshared home directory.
- Turn on `NCCL_NET_PLUGIN=ofi` to use EFA on AWS

```bash
STORAGE_ROOT=/fsx/xhuan192

# HF cache directory
HF_HOME=${STORAGE_ROOT}/shared/hf

# Keep the uv cache on /fsx so hardlink mode works with project envs there.
UV_CACHE_DIR=${STORAGE_ROOT}/.cache/uv
UV_PYTHON_CACHE_DIR=${STORAGE_ROOT}/.cache/uv/python
UV_PYTHON_INSTALL_DIR=${STORAGE_ROOT}/.local/share/uv/python

# Project storage root
PROJECT_STORAGE_ROOT=${STORAGE_ROOT}/projects/veomni

HF_TOKEN=hf_your_token_here
WANDB_API_KEY=wandb_your_api_key_here
# WANDB_PROJECT=lmm-attn-vis-v2
# WANDB_MODE=online # wandb mode: online, disabled, etc.
WANDB_ENTITY=xk-huang # your wandb username

# Use EFA (IB) on AWS for dist training
NCCL_NET_PLUGIN=ofi
```

---

Install uv to shared file system

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv self update 0.9.8 # The uv version is updated per month

set -a && source .env && set +a
# Make sure cache dir is on the same FS
uv cache dir

# Install extra video to avoid missing `av` during imports
# The text only ver imports multimodal chat template.
uv sync --locked  --extra gpu --extra video

# Double check python file
ls -l .venv/bin/python
```

Install tools via conda and use conda

- Installing ffmpeg works only if we add the library to `LD_LIBRARY_PATH`. As `torchcodec` requires `libtorchcodec_core*.so`
    
    To use conda:
    
    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ~/conda-usr
    
    conda install -c conda-forge ffmpeg

    export LD_LIBRARY_PATH="$HOME/conda-usr/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
    
    bash -c "$(curl -fsSL https://raw.githubusercontent.com/xk-huang/dotfiles/main/scripts/setup_env_shell.sh)"
    ```
    

---

Patch the `train.sh` , add `--rdzv-backend=c10d` , otherwise on AWS it hangs.

```bash
torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  --rdzv-backend=c10d \
  $additional_args $@ 2>&1 | tee log.txt
```

## Qwen3 Text example

### Download data and model

```bash
set -a && source .env && set +a
source .venv/bin/activate

# Data
hf download HuggingFaceFW/fineweb --repo-type dataset --local-dir "${PROJECT_STORAGE_ROOT}"/datasets/fineweb/ --include "sample/10BT/*"

hf download allenai/tulu-3-sft-mixture --repo-type dataset --local-dir "${PROJECT_STORAGE_ROOT}"/datasets/tulu-3-sft-mixture/ --include "data/train-00000-of-00006.parquet"

# Model
hf download Qwen/Qwen3-4B-Instruct-2507 --repo-type model --local-dir "${PROJECT_STORAGE_ROOT}"/models/Qwen/Qwen3-4B-Instruct-2507
```

### Run the code

Single node

```bash
# Single node, single GPU
source .venv/bin/activate
set -a && source .env && set +a

# Qwen2.5 example
# env \
# NNODES=1 \
# NPROC_PER_NODE=8 \
# NODE_RANK=0 \
# MASTER_ADDR=2x8xa100-80gb-0.svc-2x8xa100-80gb \
bash train.sh tasks/train_text.py configs/text/qwen2_5.yaml \
--model.model_path "${PROJECT_STORAGE_ROOT}"/models/Qwen/Qwen3-4B-Instruct-2507 \
--data.train_path "${PROJECT_STORAGE_ROOT}"/datasets/fineweb/sample/10BT \
--train.checkpoint.output_dir "${PROJECT_STORAGE_ROOT}"/runs/Qwen3-4B-Instruct-2507-CT \
--train.wandb.enable True


# Qwen3 example
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
--model.model_path "${PROJECT_STORAGE_ROOT}"/models/Qwen/Qwen3-4B-Instruct-2507 \
--data.train_path "${PROJECT_STORAGE_ROOT}"/datasets/tulu-3-sft-mixture/data \
--train.checkpoint.output_dir "${PROJECT_STORAGE_ROOT}"/runs/Qwen3-4B-Instruct-2507-sft-tulu \
--train.wandb.enable True \
--train.wandb.name Qwen3-4B-Instruct-2507-sft-tulu \
--train.accelerator.fsdp_config.fsdp_mode fsdp2 \
--train.init_device meta


# Single node, single GPU cannot run:
# init_device: meta only supports in FSDP
source .venv/bin/activate
set -a && source .env && set +a

env \
NPROC_PER_NODE=1 \
bash train.sh tasks/train_text.py configs/text/qwen2_5.yaml \
--model.model_path "${PROJECT_STORAGE_ROOT}"/models/Qwen/Qwen3-4B-Instruct-2507 \
--data.train_path "${PROJECT_STORAGE_ROOT}"/datasets/fineweb/sample/10BT \
--train.checkpoint.output_dir "${PROJECT_STORAGE_ROOT}"/runs/Qwen3-4B-Instruct-2507-CT \
--train.wandb.enable True
```

Multi node

```bash
# Multi node

# Node 0
source .venv/bin/activate
set -a && source .env && set +a

env \
NNODES=2 \
NPROC_PER_NODE=8 \
NODE_RANK=0 \
MASTER_ADDR=2x8xa100-80gb-0.svc-2x8xa100-80gb \
MASTER_PORT=25900 \
bash train.sh tasks/train_text.py configs/text/qwen2_5.yaml \
--model.model_path "${PROJECT_STORAGE_ROOT}"/models/Qwen/Qwen3-4B-Instruct-2507 \
--data.train_path "${PROJECT_STORAGE_ROOT}"/datasets/fineweb/sample/10BT \
--train.checkpoint.output_dir "${PROJECT_STORAGE_ROOT}"/runs/Qwen3-4B-Instruct-2507-CT \
--train.wandb.enable True

# Node 1
source .venv/bin/activate
set -a && source .env && set +a

env \
NNODES=2 \
NPROC_PER_NODE=8 \
NODE_RANK=1 \
MASTER_ADDR=2x8xa100-80gb-0.svc-2x8xa100-80gb \
MASTER_PORT=25900 \
bash train.sh tasks/train_text.py configs/text/qwen2_5.yaml \
--model.model_path "${PROJECT_STORAGE_ROOT}"/models/Qwen/Qwen3-4B-Instruct-2507 \
--data.train_path "${PROJECT_STORAGE_ROOT}"/datasets/fineweb/sample/10BT \
--train.checkpoint.output_dir "${PROJECT_STORAGE_ROOT}"/runs/Qwen3-4B-Instruct-2507-CT \
--train.wandb.enable True
```

# Debug distributed saving

Let’s find the difference!

- NOTE: use `set -x` to print each expanded shell command. Useful for debugging.
- NOTE: the training progress on the two nodes should be the same. If there is a lag between nodes, which means they are not synced.

Before

```bash
source .venv/bin/activate
set -a && source .env && set +a

NNODES=2
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_ADDR=2x8xa100-80gb-0.svc-2x8xa100-80gb
bash train.sh tasks/train_text.py configs/text/qwen2_5.yaml
```

After

```bash
source .venv/bin/activate
set -a && source .env && set +a

env \
NNODES=2 \
NPROC_PER_NODE=8 \
NODE_RANK=0 \
MASTER_ADDR=2x8xa100-80gb-0.svc-2x8xa100-80gb \
bash train.sh tasks/train_text.py configs/text/qwen2_5.yaml
```

# AWS EFA training speed

Run `qwen2_5.yaml`

- w/o EFA: 92.56s/it, 90.77s/it
- w/ EFA: 3.74s/it

#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.11 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

pip install -U pip setuptools wheel

# init a raw torch to avoid installation errors.
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# for fast attn
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# install base requirements
pip install -U pip setuptools wheel
pip install --index-url https://pypi.org/simple --only-binary=:all: "pyarrow>=16,<17"
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install -e .

# install additional requirements for other modules
pip install easydict==1.13
pip install sentence-transformers
pip install fire
pip install transformers==4.45.2
pip cache purge
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

pip install diffusers==0.34.0
pip install mmcv==1.7.2 --no-build-isolation --no-deps
pip install pyiqa
pip install torchmetrics
pip install streamlit
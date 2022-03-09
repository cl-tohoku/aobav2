#!/bin/bash

set -ev

pip install pip-tools
pip-compile requirements.in
pip-sync

cd pytorch-pretrained-BERT
pip install -e .
cd ..

git clone https://github.com/NVIDIA/apex
cd apex
rm -rf .git/
chmod -R 777 .
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" ./
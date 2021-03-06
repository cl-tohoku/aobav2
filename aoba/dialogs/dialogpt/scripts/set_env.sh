#!/bin/bash

set -ev

pip install pip-tools
pip-compile requirements.in
pip-sync

cd pytorch-pretrained-BERT
pip install -e .
cd ..

pip install -v --disable-pip-version-check --no-cache-dir --global-option=--deprecated_fused_adam --global-option=--cpp_ext --global-option=--cuda_ext 'git+https://github.com/NVIDIA/apex.git' --user

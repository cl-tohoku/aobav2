#!/bin/bash

conda create -n aobav2 python=3.8 -y
pyenv local miniconda3-4.0.5/envs/aobav2

pip install pip-tools
pip-compile requirements.in
pip-sync

python3 -m spacy download en_core_web_sm

cd aoba/dialogs/dialogpt/pytorch-pretrained-BERT
pip install -e .
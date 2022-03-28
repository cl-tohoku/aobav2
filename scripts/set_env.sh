#!/bin/bash

conda create -n aobav2 python=3.8 -y
pyenv local miniconda3-4.0.5/envs/aobav2

pip install pip-tools
pip-compile requirements.in
pip-sync

python3 -m spacy download en_core_web_sm

git clone https://github.com/pytorch/fairseq -b 1.0.0a0 lib/fairseq
cd lib/fairseq
git checkout -b 8adff65ab30dd5f3a3589315bbc1fafad52943e7
pip install -e .
python setup.py build_ext --inplace
cd ../../

cd aoba/dialogs/dialogpt/pytorch-pretrained-BERT
pip install -e .



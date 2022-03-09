#!/bin/bash
# USAGE: bash $0 {PYENV_NAME}

PATH=$HOME/.linuxbrew/bin:$HOME/.linuxbrew/sbin:$PATH

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

pyenv activate ${PYENV_NAME}

source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn/7.6/7.6.5 gcc/7.4.0

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

DATE=`date +%Y%m%d-%H%M`
echo $DATE

hostname
uname -a
which python
python --version
pip list

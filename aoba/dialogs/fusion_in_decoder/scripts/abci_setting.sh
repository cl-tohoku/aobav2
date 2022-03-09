ENVIRONMENT=$1
PATH=$HOME/.linuxbrew/bin:$HOME/.linuxbrew/sbin:$PATH

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv init --path)"

pyenv activate $ENVIRONMENT

source /etc/profile.d/modules.sh
module load cuda/11.1/11.1.1 nccl/2.9/2.9.9-1 gcc/7.4.0 cudnn/8.2/8.2.0

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

DATE=`date +%Y%m%d-%H%M`
echo $DATE

hostname
uname -a
which python
python --version
pip list
# USAGE: source $0 {pyenv name}

PATH=$HOME/.linuxbrew/bin:$HOME/.linuxbrew/sbin:$PATH

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate $1

source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89  # load cuda
module load gcc/7.4.0  # load gcc

export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8

DATE=`date +%Y%m%d-%H%M`
echo $DATE

hostname
uname -a
which python
python --version
pip list
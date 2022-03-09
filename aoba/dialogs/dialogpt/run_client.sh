#!/bin/bash

USAGE="bash $0 <HOST>"

if [ -z $1 ] ; then
  echo $USAGE
  IP=$(hostname -I | awk -F ' ' '{print $1}')
  echo "probably ... ${IP}"
  exit 1
fi

DIR_MODEL=/work02/SLUD2021/models/dialogpt

python ./client_dialogpt.py \
  --dgpt_model ${DIR_MODEL}/GP2-pretrain-step-300000.pkl \
  --dgpt_config ${DIR_MODEL}/config.json \
  --dgpt_toker ${DIR_MODEL}/tokenizer \
  --dgpt_max_history 1 \
  --host $1 \
  --port 45000

#!/bin/bash

USAGE="bash $0 <HOST>"

if [ -z $1 ] ; then
  echo $USAGE
  IP=$(hostname -I | awk -F ' ' '{print $1}')
  echo "probably ... ${IP}"
  exit 1
fi

python ./client_ilys_fairseq.py /work02/SLUD2021/datasets/dialogue/ja/data_fairseq_preprocess/fairseq_phase2 \
  --path /work02/SLUD2021/models/fairseq_only_phase2/checkpoint_best.pt \
  --beam 20 \
  --max-len-b 30 \
  --min-len 5 \
  --source-lang context \
  --target-lang response \
  --no-repeat-ngram-size 2 \
  --diverse-beam-groups 5 \
  --diverse-beam-strength 2.0 \
  --nbest 10 \
  --host $1

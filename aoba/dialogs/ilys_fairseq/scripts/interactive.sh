#!/bin/bash

USAGE="bash $0"


python ./bot_ilys_model.py /work02/SLUD2021/datasets/dialogue/ja/data_fairseq_preprocess/fairseq_phase2 \
  --path /work02/SLUD2021/models/fairseq_only_phase2/checkpoint_best.pt \
  --beam 20 \
  --max-len-b 30 \
  --min-len 5 \
  --source-lang context \
  --target-lang response \
  --no-repeat-ngram-size 2 \
  --diverse-beam-groups 5 \
  --diverse-beam-strength 2.0 \
  --nbest 10

#!/bin/bash

set -e

YEAR=$1
FI_CONTEXT=$2
FI_RESPONSE=$3

DEST=/work/miyawaki/datasets/SLUD2021/twitter/processed/$YEAR
FO_CONTEXT="$DEST/processed_$(basename $FI_CONTEXT)"
FO_RESPONSE="$DEST/processed_$(basename $FI_RESPONSE)"

python prepro_twitter.py \
  --fi_context $FI_CONTEXT \
  --fi_response $FI_RESPONSE \
  --fo_context $FO_CONTEXT \
  --fo_response $FO_RESPONSE \
  --fi_config prepro/params_prepro.yml
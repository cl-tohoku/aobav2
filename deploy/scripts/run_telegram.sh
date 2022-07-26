#!/bin/bash

set -ex
USAGE="bash $0 ${API_TOKEN}"

export PYTHONPATH="../:$PYTHONPATH"


API_TOKEN=$1
HOST=$(hostname -I | awk -F ' ' '{print $1}')

python run_telegram.py \
  common.api_token=${API_TOKEN} \
  common.host="${HOST}" \
  common.dialogue_agent="telegrams.dialogue_agents.BaseDialogueAgent" \
  aobav1.port=42000 \
  dialogpt.port=45000 \
  fid.port=50000 \
  nttcs.port=40000

#!/bin/bash

USAGE="bash $0 <HOST>"

if [ -z $1 ] ; then
  echo $USAGE
  IP=$(hostname -I | awk -F ' ' '{print $1}')
  echo "probably ... ${IP}"
  exit 1
fi

python ./client_fid.py \
  --fid_config configs/client_fid.json \
  --host $1 \
  --port 50000

#!/bin/bash

USAGE="bash $0 [qsub]"

PYENV_NAME=$(pyenv version | awk -F ' ' '{print $1}')

__DIR__=$(cd $(dirname $0); pwd)
__FILE__=$(basename $0)

DIR_DATA=/groups2/gcb50246/miyawaki/data/slud2021
FI_TRAIN_CON=$DIR_DATA/train_10K.context
FI_TRAIN_RES=$DIR_DATA/train_10K.response
FI_VALID_CON=$DIR_DATA/valid_full.context
FI_VALID_RES=$DIR_DATA/valid_full.response
DEST=/groups2/gcb50246/ariyama/SLUD2021/DialoGPT/


if [ "$1" = "qsub" ] ; then
    echo "qsub"
    qsub -cwd -g gcb50246 \
        -N .${__FILE__} -j y \
        -l rt_G.small=1 -l h_rt=168:00:00 \
        ${__DIR__}/${__FILE__}

else
    source scripts/abci_setting.sh $PYENV_NAME
    
    echo "===== Create tsv file ====="
    bash scripts/merge_file.sh ${FI_TRAIN_CON} ${FI_TRAIN_RES} ${DEST}/train_10K.tsv
    bash scripts/merge_file.sh ${FI_VALID_CON} ${FI_VALID_RES} ${DEST}/valid.tsv

    echo "===== Create data db ====="
    bash scripts/create_db.sh ${DEST}/train_10K.tsv 128
    ls ${DEST}

fi
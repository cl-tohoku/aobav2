#!/bin/bash

USAGE="bash $0 [fi_tsv] [max_length]"

fi_tsv=$1
[ -z $2 ] && max_length=128 || max_length=$2

if [ -f $fi_tsv ] ; then
    python prepro.py \
        --corpus $fi_tsv \
        --max_seq_len $max_length \
        --ja
else
    echo $USAGE
    exit 1
fi
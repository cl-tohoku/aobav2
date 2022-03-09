#!/bin/bash

USAGE="bash $0 [fi_context] [fi_response] [fo_merge]"

fi_con=$1
fi_res=$2
fo_merge=$3

if [ -f $fi_con ] && [ -f $fi_res ] && [ ! -f $fo_merge ] ; then
    mkdir -p $(dirname $fo_merge)
    paste -d "\t" $fi_con $fi_res | awk -F '\t' '{print "0.0 "$1"\t1.0 "$2}' > $fo_merge
    echo "write ... ${fo_merge}"
elif [ -f $fo_merge ] ; then
    echo "exist ... ${fo_merge}"
else
    echo $USAGE
    exit 1
fi

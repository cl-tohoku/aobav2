#!/bin/bash
# USAGE: bash $0 {fi_src} {fi_tgt} {dest}

SRC=$1
TGT=$2
DEST=$3/${1%.*}_${2%.*}

mkdir -p $DEST

ROOT=`pwd`
GIZA=$ROOT/giza-pp/GIZA++-v2

cd $DEST
$GIZA/plain2snt.out $SRC $TGT
$GIZA/snt2cooc.out $SRC.vcb $TGT.vcb ${SRC}_${TGT}.snt > ${SRC}_${TGT}.cooc
$GIZA/GIZA++ -s $SRC.vcb -t $TGT.vcb -c ${SRC}_${TGT}.snt -coocurrence ${SRC}_${TGT}.cooc -outputpath $DEST
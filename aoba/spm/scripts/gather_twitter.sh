#!/bin/sh
# spm を学習するための Twitter データを収集

DDIR=/work/miyawaki/datasets/SLUD2021/twitter
DEST=/work/miyawaki/datasets/SLUD2021/twitter/spm
FO_SPM=$DEST/shuffled_twitter_context_8M.txt

mkdir -p $DEST

if [ ! -f $FO_SPM ] ; then
  # 各年度ごとにランダムに文を選択
  if [ ! -d $DEST ] ; then
    for ((y=2013; y<=2020; y++)) ; do
      shuf -n 1000000 ${DDIR}/${y}/processed.context > ${DEST}/shuffled_twitter_context_${y}.txt
      wc -l ${DEST}/shuffled_twitter_context_${y}.txt
    done
  fi
  # 選択したファイルを一つのファイルにまとめる
  cat $DEST/*.txt > $FO_SPM
fi

wc -l $FO_SPM

#!/bin/sh

MODEL=/work02/ariyama/slud2021/spm/8M_tweets.cr9999.bpe.32000.model
DDIR=/work02/SLUD2021/datasets/dialogue/ja
PYDIR=/home/ariyama/SLUD2021/src/parsers/spm

TARGETS=(
  "${DDIR}/dailydialog"
  "${DDIR}/empathetic_dialogues"
  "${DDIR}/wizard_of_wikipedia"
  "${DDIR}/convai2"
  "${DDIR}/blended_skill_talk"
)

for directory in ${TARGETS[@]} ; do
  for file in ${directory}/processed_* ; do
    DEST=`basename ${file}`
    python ${PYDIR}/encode_spm.py --model $MODEL < ${file} > ${directory}/spm_${DEST}
    sed -i 's/< s >/<s>/g' ${directory}/spm_${DEST}
  done
done

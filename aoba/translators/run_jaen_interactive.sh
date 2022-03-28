#!/usr/bin/env bash
# This script is copied from https://github.com/cl-tohoku/wmt2020-resources

set -xe

INPUT=${1}

TMP_DIR="tmp_jaen"
mkdir -p ${TMP_DIR}

# Paths
TC_SCRIPT="./mosesdecoder/scripts/recaser/truecase.perl"
DETRUE_SCRIPT="./mosesdecoder/scripts/recaser/detruecase.perl"
SPM_MODEL_EN="/work01/kiyono/wmt2020/share/bpe_models/vocab-32000_coverage_10000.en.model"
SPM_MODEL_JA="/work01/kiyono/wmt2020/share/bpe_models/vocab-32000_coverage_09998.ja.model"
DATA_DIR="/work01/kiyono/wmt2020/share/for_finetuning/fairseq_data/ja-en/vocab_32000_clean_parallel_tc_for_tag_redo"
MODEL="/work01/kiyono/wmt2020/share/final_models/jaen/seed_1010_l2r/average_checkpoints/checkpoint_goal_update_80000_average_10.pt"

# Preprocessing
## 1. Subword splitting using sentencepiece w/ BPE algorithm
python subword_encode.py --model ${SPM_MODEL_JA} < ${INPUT} > ${TMP_DIR}/$(basename $INPUT).spm

# Interactive Decoding
## You might want to tweak buffer-size, batch-size, beam and lenpen if necessary
head ${TMP_DIR}/$(basename $INPUT).spm |\
    fairseq-interactive ${DATA_DIR} \
    --path ${MODEL} \
    --source-lang ja_bpe \
    --target-lang en_bpe \
    --buffer-size 100 \
    --batch-size 2 \
    --log-format simple \
    --beam 1 \
    --lenpen 1.0 \
    | tee ${TMP_DIR}/translation.log

# Postprocessing
## 1. Sanitize fairseq output
cat ${TMP_DIR}/translation.log | grep -e "^H" | cut -f1,3 | sed 's/^..//' | sort -n -k1  | cut -f2 > ${TMP_DIR}/translation.clean
## 2. Convert subword sequence to word sequence
python subword_decode.py ${SPM_MODEL_EN} < ${TMP_DIR}/translation.clean > ${TMP_DIR}/translation.clean.detok
## 3. Apply detruecaser
perl ${DETRUE_SCRIPT} < ${TMP_DIR}/translation.clean.detok > ${TMP_DIR}/translation.clean.detok.detrue



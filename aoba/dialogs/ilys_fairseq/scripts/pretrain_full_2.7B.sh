#! /bin/sh

ROOT_ILYS=`pwd`  # 実行時のディレクトリ
DATE=`date +%Y%m%d-%H%M`

END="\e[m"
GREEN="\e[32m"
BLUE="\e[34m"
YELLOW="\e[33m"


# parametes =======================================

PYENV_NAME="miniconda3-3.19.0/envs/ilys"

DIR_DATA="/groups2/gcb50246/miyawaki/data/fairseq_preprocess/pretrain_full"
DIR_OUT="/groups2/gcb50246/miyawaki/data/fairseq_train/pretrain_full_2.7B"
DIR_MODEL="${DIR_OUT}/models"
DIR_TENSORBOARD="${DIR_OUT}/tensorboards"

mkdir -p ${DIR_MODEL} ${DIR_TENSORBOARD}

ENC_EMB=2560  # 1024
ENC_FFN=10240 # 8192
ENC_LAYER=2
ENC_HEAD=32
DEC_EMB=2560
DEC_FFN=10240
DEC_LAYER=24  # 16
DEC_HEAD=32

UPDATE_FREQ=2 #16


# pretrain =========================================

if [ "$1" == "qsub" ] ; then
    qsub -cwd -g gcb50246 \
        -N .log_pretrain_2.7B -j y \ 
        -l rt_G.large=1 -l h_rt=72:00:00 \
        ${ROOT_ILYS}/$0

else

    source ${ROOT_ILYS}/scripts/abci_setting.sh

    CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ${DIR_DATA} \
        --seed 2021 \
        --source-lang "context" \
        --target-lang "response" \
        --arch "transformer_vaswani_wmt_en_de_big" \
        --activation-fn "gelu" \
        --dropout 0.1 \
        --attention-dropout 0.0 \
        --relu-dropout 0.0 \
        --encoder-embed-dim ${ENC_EMB} \
        --encoder-ffn-embed-dim ${ENC_FFN} \
        --encoder-layers ${ENC_LAYER} \
        --encoder-attention-heads ${ENC_HEAD} \
        --encoder-normalize-before \
        --decoder-embed-dim ${DEC_EMB} \
        --decoder-ffn-embed-dim ${DEC_FFN} \
        --decoder-layers ${DEC_LAYER} \
        --decoder-attention-heads ${DEC_HEAD} \
        --decoder-normalize-before \
        --share-all-embeddings \
        --max-tokens 2000 \
        --max-update 400000 \
        --update-freq ${UPDATE_FREQ} \
        --num-workers 15 \
        --fp16 \
        --optimizer adam \
        --adam-betas "(0.9, 0.98)" \
        --lr-scheduler "inverse_sqrt" \
        --warmup-updates 3125 \
        --warmup-init-lr 1e-07 \
        --lr 1e-03 \
        --min-lr 1e-09 \
        --weight-decay 0.0 \
        --clip-norm 0.1 \
        --criterion "label_smoothed_cross_entropy" \
        --label-smoothing 0.1 \
        --log-format "simple" \
        --tensorboard-logdir ${DIR_TENSORBOARD} \
        --keep-last-epochs 1 \
        --keep-interval-updates 10 \
        --save-interval-updates 10000 \
        --log-interval 2500

    DATE=`date +%Y%m%d-%H%M`
    echo $DATE

fi
#!/bin/bash
# https://unidic.ninjal.ac.jp/
# 出力形式は ${DEST/dicrc を参照
# node-format-unidic22 = %m\t%f[0],%f[1],%f[2],%f[3],%f[4],%f[5],%f[6],%f[7],%f[8],%f[9],%f[10],%f[11],%f[12],"%f[13]","%f[14]","%f[15]","%f[16]","%f[17]","%f[18]",%f[19],%f[20],%f[21],%f[22],%f[23],"%f[24]","%f[25]","%f[26]",%f[27],%f[28]\n

DEST=/work02/SLUD2021/datasets

wget -nc https://unidic.ninjal.ac.jp/unidic_archive/csj/3.1.0/unidic-csj-3.1.0.zip -p $DEST \
    && unzip $DEST/unidic-csj-3.1.0.zip -d $DEST \
    && rm -rf $DEST/unidic-csj-3.1.0.zip
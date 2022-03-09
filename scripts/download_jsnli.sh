#!/bin/bash

DEST=/work02/SLUD2021/datasets/snli
mkdir -p $DEST

if [ ! -d $DEST/jsnli_1.1 ] ; then
  wget -nc https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip -O $DEST/jsnli_1.1.zip \
    && unzip $DEST/jsnli_1.1.zip -d $DEST \
    && rm $DEST/jsnli_1.1.zip
fi
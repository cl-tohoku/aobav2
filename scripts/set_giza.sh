#!/bin/bash

cd lib/
if [ ! -d giza-pp ] ; then
  git clone git@github.com:moses-smt/giza-pp.git \
    && cd giza-pp \
    && make
fi
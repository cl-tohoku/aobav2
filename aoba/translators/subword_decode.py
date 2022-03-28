# -*- coding: utf-8 -*-
# This script is copied from https://github.com/cl-tohoku/wmt2020-resources

import sys

import sentencepiece


def main(fi, model_path):
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(model_path)
    for line in fi:
        print(sp.DecodePieces(line.strip().split(' ')))


if __name__ == "__main__":
    main(sys.stdin, sys.argv[1])
# -*- coding: utf-8 -*-
# This script is copied from https://github.com/cl-tohoku/wmt2020-resources

import argparse
import os
import sys

import sentencepiece


def get_args():
    parser = argparse.ArgumentParser(description='my script')
    parser.add_argument('--model', '-m', required=True, type=os.path.abspath, help='file to model')
    args = parser.parse_args()
    return args


def main(fi, model_path):
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(model_path)

    for line in fi:
        print(' '.join(sp.EncodeAsPieces(line.strip())))


if __name__ == "__main__":
    args = get_args()
    main(sys.stdin, args.model)
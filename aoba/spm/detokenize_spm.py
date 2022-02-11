import argparse
from os.path import abspath
import sys

from sentencepiece import SentencePieceProcessor


def main(lines):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=abspath, help="Path to model file")
    args = parser.parse_args()

    sp = SentencePieceProcessor()
    sp.Load(args.model)

    for line in lines:
        print(sp.DecodePieces(line.strip().split(" ")))



if __name__ == "__main__":
    main(sys.stdin)

import argparse
from os.path import abspath
import sys

from sentencepiece import SentencePieceProcessor


def main(lines):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=abspath, help="Path to model file")
    parser.add_argument("--sep", default=" ")
    args = parser.parse_args()

    sp = SentencePieceProcessor()
    sp.Load(args.model)

    for line in lines:
        print(args.sep.join(sp.EncodeAsPieces(line.rstrip("\n"))))



if __name__ == "__main__":
    main(sys.stdin)

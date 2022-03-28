from enum import Enum

import sentencepiece


class SpecialToken(Enum):
    COT = "<ST1>"
    OOGIRI_1 = "<ST3>"
    OOGIRI_2 = "<ST4>"
    OOGIRI_3 = "<ST5>"


class SpmTokenizer:
    def __init__(self, spm_path: str):
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(spm_path)

    def encode(self, text: str) -> str:
        return ' '.join(self.sp.EncodeAsPieces(text))

    def decode(self, text: str) -> str:
        return self.sp.DecodePieces(text.strip().split(' '))

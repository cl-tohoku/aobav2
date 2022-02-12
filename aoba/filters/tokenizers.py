import dataclasses
import re
from typing import List

import MeCab


@dataclasses.dataclass
class MecabField:
    """ 形式は dicrc 参照 """
    tid: int        # token ID
    surface: str    # 表層形
    pos1: str       # 品詞
    pos2: str       # 品詞細分類1
    pos3: str       # 品詞細分類2
    pos4: str       # 品詞細分類3
    cType: str      # 活用型
    cForm: str      # 活用形
    lemma: str      # 原形
    lForm: str = dataclasses.field(default='')     # 読み
    pron: str = dataclasses.field(default='')       # 発音
    def __getitem__(self, item):
        return getattr(self, item)


class MecabParser(object):
    def __init__(self):
        dict = "/work02/SLUD2021/datasets/unidic-csj-3.1.0"
        self.tagger = MeCab.Tagger(f"-d {dict}")
    
    def __doc__(self):
        return "https://taku910.github.io/mecab/"

    def __call__(self, text:str) -> List[MecabField]:
        text = text.replace(",", "、")
        parsed_text = self.tagger.parse(text).rstrip("EOS\n").split("\n")
        return list(map(lambda x: MecabField(*re.split("[\t,]", x)), parsed_text))

    def select(self, parsed_text:List[MecabField], **kwargs) -> List[MecabField]:
        return list(filter(lambda x: all(x[field] in values for field, values in kwargs.items()), parsed_text))



if __name__ == "__main__":
    from pprint import pprint
    text = "吾輩は猫である。名前はまだない。"
    parser = MecabParser()
    pprint(parser(text))
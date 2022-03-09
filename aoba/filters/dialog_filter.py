from os.path import isfile
import re
from typing import List
import unicodedata

Token = str
Utterance = List[Token]
Dialog = List[Utterance]


class DialogFilter:
    def __init__(self, fi_ng_word="data/ng_words.txt", min_v=6, max_v=29, kana_ratio_thrs=0.3, _parentheses=True):
        self.min_v = min_v
        self.max_v = max_v
        self.kana_ratio_thrs = kana_ratio_thrs
        self._parentheses = _parentheses
        self.ng_words = set([line.strip() for line in open(fi_ng_word)]) if fi_ng_word is not None else []

    def __call__(self, dialog:Dialog):
        """ Dialog 単位で、フィルタリング対象かどうか判定する """
        if (self.min_v and self.max_v) and self.exceed_word_count(dialog):
            return True
        if (self.ng_words) and self.include_ng_word(dialog):
            return True
        if (self._parentheses) and self.include_parentheses(dialog):
            return True
        if (self.kana_ratio_thrs) and self.is_few_kana(dialog):
            return True
        return False

    @staticmethod
    def is_kana(ch):
        try:
            return unicodedata.name(ch).startswith(("HIRAGANA LETTER", "KATAKANA LETTER"))
        except ValueError:
            return False

    # ひらがな・カタカナの割合が 30 ％以下であれば除去
    def is_few_kana(self, dialog):
        for utter in dialog:
            row_text: str = "".join(utter)
            n_char = len(row_text)
            kana_ratio = sum(self.is_kana(c) for c in row_text) / n_char
            if n_char == 0 or kana_ratio <= self.kana_ratio_thrs:
                return True

    # 不適切な単語が含まれる発話を除去
    def include_ng_word(self, dialog):
        for utter in dialog:
            for token in utter:
                if token in self.ng_words:
                    return True

    # カッコが含まれる発話を除去
    def include_parentheses(self, dialog):
        for utter in dialog:
            if re.search("\(.+?\)", "".join(utter)) or re.search("（.+?）", "".join(utter)):
                return True

    # 1発話の単語数が min_v ~ max_v に収まらないものは除外
    def exceed_word_count(self, dialog):
        for utter in dialog:
            if len(utter) < self.min_v or self.max_v < len(utter):
                return True



if __name__ == "__main__":
    dialog_filter = DialogFilter()
    dialog = [utter.split() for utter in ["今日 は いい 天気 です ね 。", "そう だ ね"]]
    is_filter = dialog_filter(dialog)
    print(is_filter)
from os.path import dirname, join
import sys
import re
import unicodedata

import emoji
import mojimoji
import wikitextparser

sys.path.append(dirname(__file__))
from parsers import MecabParser


class SentenceNormalizer:
    def __init__(self):
        self.emoji = emoji.UNICODE_EMOJI["en"]
        self.parser = MecabParser()

    def __call__(self, text):
        text = self.normalize(text)
        text = self.parentheses(text)
        text = self.normalize_with_mecab(text)
        return text

    def normalize(self, text):
        text = unicodedata.normalize("NFKC", text)
        # special character
        text = re.sub("[\uA000-\uF8FF|\uFF3E|\u002A|\u005E|'\u0020']+", "", text)
        text = re.sub("[\u00C0-\u2E41]+", "", text)
        text = re.sub(r"(\.)+", "。", text)
        # emoji
        text = "".join([c for c in text if c not in self.emoji])
        return text

    @staticmethod
    def parentheses(text, patterns=["\(.+?\)", "\<.+?\>", "\[.+?]", "（.+?）"]):
        text = text.replace("<s>", "@@s@@")
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        return text.replace("@@s@@", "<s>")

    def normalize_with_mecab(self, text):
        previous_word = ""
        output_text = []
        for meta in self.parser(text):
            word = meta.surface
            # 1単語内で3文字以上続く場合は2文字にまとめる
            word = re.sub(r"(.){2, }", "\g<1>"*2, word)
            # 1単語が"笑","w"の繰り返しの場合は取り除く
            word = re.sub(r"^(笑|w)*$", "", word)
            # 同じ単語が2回以上続く場合は1回にする
            next_word = word
            if next_word != previous_word:
                output_text.append(next_word)
                previous_word = next_word
        return "".join(output_text)

    def normalize_wikidump(self, text):
        # 内部リンク・ファイル・カテゴリの除去  https://qiita.com/FukuharaYohei/items/95a9ceaaf60858c3e483
        text = re.sub(r"""
            \[\[             # "[["(マークアップ開始)
            (?:              # キャプチャ対象外のグループ開始
                [^|]*?       # "|"以外の文字0文字以上、非貪欲
                \|           # "|"
            )*?              # グループ終了、このグループが0以上出現、非貪欲(No27との変更点)
            (                # グループ開始、キャプチャ対象
              (?!Category:)  # 否定の先読(含んだ場合は対象外としている)
              ([^|]*?)    # "|"以外が0文字以上、非貪欲(表示対象の文字列)
            )
            \]\]        # "]]"（マークアップ終了）        
            """, r"\1", text, flags=re.MULTILINE + re.VERBOSE)

        # 除去対象：{{lang|言語タグ|文字列}}
        text = re.sub(r"""
            \{\{lang    # "{{lang"(マークアップ開始)
            (?:         # キャプチャ対象外のグループ開始
                [^|]*?  # "|"以外の文字が0文字以上、非貪欲
                \|      # "|"
            )*?         # グループ終了、このグループが0以上出現、非貪欲
            ([^|]*?)    # キャプチャ対象、"|"以外が0文字以上、非貪欲(表示対象の文字列)
            \}\}        # "}}"(マークアップ終了)
            """, r"\1", text, flags=re.MULTILINE + re.VERBOSE)

        text = re.sub(r"('{5}|'{2,3})(.*?)('{5}|'{2,3})", r"\2", text)  # 強調の削除
        text = re.sub(r"(==+)(.*?)(==+)", "", text)  # 見出し
        text = re.sub(r"\((.*?)\)", "", text)  # "()"で書かれるパターンを括弧を含めて取り除く
        text = re.sub(r"(\(|（).*?(）|\))", "", text)  # 日本語のフォントの全角括弧（）も取り除く
        text = re.sub(r"<!--(.*?)-->", "", text)  # コメントの削除
        text = re.sub(r"<.*>", "", text)  # htmlタグ系
        text = re.sub(r"\{\|class=.*?\|\}", "", text, flags=re.DOTALL)
        text = re.sub(r"(^(\*|#|;|:)+).*?$", "", text, flags=re.MULTILINE)  # 箇条書きを取り除く
        text = re.sub(r"\[(http|https):\/\/(.*?)\]", "", text)  # 外部リンクの除去
        text = re.sub(r"(http|https):\/\/(.*)(.com|.org|.net|.int|.edu|.gov|.mil|.jp|.html)", "", text)  # 外部リンクの除去その2(他にもよく使われるドメインがあったら追記してください)
        text = re.sub(r"#REDIRECT \[\[(.*?)\]\]", "", text)  # リダイレクトの削除
        text = re.sub(r"~~~~", "", text)  # 署名の削除
        text = re.sub(r"\{\{.*\}\}", "", text)  # 「特によく使われるテンプレート」の削除
        text = re.sub(r"^----", "", text)  # 水平線の削除

        text = wikitextparser.remove_markup(text)
        text = re.sub("^Category:.*?$", "", text, flags=re.MULTILINE)
        return text


if __name__ == "__main__":
    text = "ﾔｯﾎｰ...（笑）笑www😄..."
    normalizer = SentenceNormalizer()
    normalized_text = normalizer(text)
    print(text)
    print(normalized_text)
# aobav2 bot


## ディレクトリ構造

```yaml
- aoba/:
    # 前処理・後処理で使用するモジュール群
    - filters/:
        - parsers.py:                       Mecab parser
        - normalizers.py:                   文正規化や Wikidump のフィルタ処理
        - bert_predictors/:
            - next_utterance_predictor.py:  NSP
            - nli_predictor.py:             JSNLI
    # sentencepiece に関するファイル群
    - spm/:
        - train_spm.py:
        - encode_spm.py:
        - detokenize_spm.py:
        - scripts/:
            - gather_twitter.sh:
            - apply_spm_to_dialog_datasets.sh:

# 前処理などで参照するデータ（学習データは置かない）
- data/:
    - ng_words.txt:                         NG 単語リスト

# 実行スクリプト
- scripts/:
    - set_*.sh:                             セットアップ関連
    - download_*.sh:                        ダウンロード関連
    - prepro_*.sh:                          前処理関連

# 前処理
- datasets.yml:                             データセットと読み込みモジュール
- prepro/:
    - convert_format.py:                    List[Dialog] の形式に変換する（Dialog = ["こんにちは", "いい天気ですね", ...]）
    - formats/:
        - base.py:
        - dailydialog.py:                   DailyDialog データ List[Dialog] の形式で読み込む
```

## filters

```py
from aoba import (
    MecabParser,
    SentenceNormalizer,
    NextUtterancePredictor, 
    JsnliPredictor,
)

# MecabParser
parser = MecabParser()
parsed_text = parser("吾輩は猫である。名前はまだない。")

# SentenceNormalizer
normalizer = SentenceNormalizer()
normalized_text = normalizer("ﾔｯﾎｰ...（笑）笑www😄...")

# NextUtterancePredictor
predictor = NextUtterancePredictor("/work01/slud_livechat_2020/mlm-checkpoint-43000-pytorch-model.bin")
results = predictor(
    ["今日はいい天気ですね", "外に遊びにいきましょう"],     # contexts
    ["いいですね", "どこいきますか？"]                   # response_candidates
)

# JsnliPredictor
predictor = JsnliPredictor("/work02/SLUD2021/github/src/submodules/jsnli/outputs/best-24000")
result = predictor([
    [
        "ワクチン打ったら副作用が辛かった。",  # premise
        "具体的にどんな副作用がありました？"   # hypothesis
    ]
])
```
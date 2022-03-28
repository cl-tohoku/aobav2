# aobav2 bot

```bash
git clone --recursive git@github.com:cl-tohoku/aobav2.git
# git submodule update --init --recursive
```

## Telegram での実行

[./deploy](./deploy) ディレクトリを参照


## ディレクトリ構造

```yaml
- aoba/:
    # 前処理・後処理で使用するモジュール群
    - filters/:
        - parsers.py:                       Mecab parser
        - normalizers.py:                   文正規化や Wikidump のフィルタ処理
        - dialog_filter.py:                 フィルタ対象か判定（単語数/NG単語/かっこ/仮名率）
        - postprocess_scorer.py:            スコアを算出（Jaccard/SIF/重複度）
        - bert_predictors/:
            - next_utterance_predictor.py:  NSP
            - nli_predictor.py:             JSNLI
    # giza+アラインメント評価に関するファイル群
    - giza/:
        - evaluate_trans.py:                アラインメント評価
        - backtrans.py:                     類似度関連の評価基準を定義
        - sbert.py:                         SentenceTransformer を使用した類似度評価（使用しない）
    # sentencepiece に関するファイル群
    - spm/:
        - train_spm.py:
        - encode_spm.py:
        - detokenize_spm.py:
        - scripts/:
            - gather_twitter.sh:
            - apply_spm_to_dialog_datasets.sh:
    # dialogs
    - dialogs/:
        - ilys_fairseq/: TBA
        - nttcs_fairseq/: TBA
        - fusion_in_decoder/: TBA
        - dialogpt/: TBA
        - wiki_template/:
            - datasets.yml:                 template_dialogue に必要なファイル
            - wiki_template_dialogue.py:    テンプレート応答
    # knowledges
    - knowledges/:
        - dpr/: TBA
        - esearch/:
            - index_config.json:
            - index_config_sudachi.json:
            - register_docs_async_icu_normalizer.py:   jsonl データを ES に登録する
            - es_search.py:                            登録した index_name を用いて検索

# 前処理などで参照するデータ（学習データは置かない）
- data/:
    - ng_words.txt:                         NG 単語リスト
    - wiki_template_dialog/:
        - context_templates.json:           テンプレート対話に必要
        - response_templates.jsonl:         テンプレート対話に必要

# 実行スクリプト
- scripts/:
    - set_*.sh:                             セットアップ関連
    - download_*.sh:                        ダウンロード関連
    - prepro_*.sh:                          前処理関連
    - register_wikidump.sh:                 wikidump を ES に登録する
    - run_giza.sh:                          GIZA++ を用いて A3 ファイルを作成

# 前処理
- datasets.yml:                             データセットと読み込みモジュール
- prepro/:
    - create_parallel_corpus.py:            .context/.response ファイルの作成

# telegram
- deploy/:
    - run_telegram.py:
    - telegrams/:

# lib
- lib/:
    - elasticsearch-7.10.0/:                scripts/set_elasticsearch.sh
    - giza-pp/:                             scripts/set_giza.sh
```

## 前処理モジュール

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

## giza

```bash
$ bash scripts/set_giza.sh
$ bash scripts/run_giza.sh {fi_src} {fi_tgt} {dest}
```

```bash
# {a3ファイル, backtransファイル} を使用したフィルタリング対象の決定
$ python aoba/giza/evaluate_trans.py \
    --a3_file {a3_file} \
    --backtrans_file {backtrans_file} \
    --output_file {output_file}
```

```py
from aoba import (
    TransEvaluator
)

# backtrans 評価
evaluator = TransEvaluator()
result = evaluator(
    "she was interested in world history because she read the book", # source
    "she read the book because she was interested in world history"  # target
)
```

## 知識モジュール

### Elastic Search

* [./aoba/knowledges/esearch/README.md](./aoba/knowledges/esearch/README.md) を参照
* ElasticStack について詳しく知りたい場合は、以下の書籍がおすすめ（python による解説は残念ながらない）
  * [ElasticStack 実践ガイド（Amazon）](https://www.amazon.co.jp/-/en/%E6%83%A3%E9%81%93-%E5%93%B2%E4%B9%9F/dp/4295009776/ref=pd_lpo_1?pd_rd_i=4295009776&psc=1)


### Dense Passage Retrieval

* https://github.com/cl-tohoku/AIO2_DPR_baseline

```python
from omegaconf import OmegaConf
from aoba import DenseExtractor

cfg = OmegaConf.load(open("aoba/knowledges/dense_passage_retrieval/interact_retriever.yml"))
dense_extractor = DenseExtractor(cfg)

query = "東京都港区芝公園にある総合電波塔の名前は何？"
retrieved_passages = dense_extractor(query, n_docs=5)
print(retrieved_passages[0])

{
    'id': 'wiki:1212368',
    'score': 44.463657,
    'title': '東京タワー',
    'text': '東京タワーは、日本の東京都港区芝公園にある総合電波塔の愛称である。正式名称は日本電波塔。創設者は前田久吉。'
}
```

## 対話モデル

### Wikipedia テンプレート応答

```py
from aoba import WikipediaTemplateDialogue

template_dialogue = WikipediaTemplateDialogue()
response = template_dialogue("東京タワーって知ってる？")
```

### Fusion-in-Decoder

```py
import argparse
from omegaconf import OmegaConf
from aoba import DenseExtractor, FidModel

cfg = OmegaConf.load(open("aoba/knowledges/dense_passage_retrieval/interact_retriever.yml"))
dense_extractor = DenseExtractor(cfg)

parser = argparse.ArgumentParser()
parser = FidModel.add_parser(parser)
args = parser.parse_args()
decoder = FidModel(args)

query = "東京都港区にある東京タワーは何の建物ですか？"
retrieved_passages = dense_extractor(query, n_docs=5)
input_data = FidModel.convert_retrieved_psgs(query, retrieved_passages)
responses = decoder(input_data)
responses[0]

"東京都港区芝公園にある総合電波塔の愛称である。正式名称は日本電波塔。創設者は前田久吉。"
```

### DialoGPT

```py
import argparse
from aoba import DialoGptModel

parser = argparse.ArgumentParser(description="")
parser = DialoGptModel.add_parser(parser)
args = parser.parse_args()

decoder = DialoGptModel(args)

history = ["おはようございます"]
response = decoder(history, num_beams=5)
responses[0]

"おはようございます。今日も一日頑張りましょう。"
```


# 前処理

## Twitter データに対するフィルタリング
- https://io-lab.esa.io/posts/2193

```bash
# 改良版（実行内容は変わっていない。実行可能かチェックが必要）
bash scripts/prepro_twitter.sh {fi_context} {fi_response} {year}
```

## 2020 (NII/LINE)

- Basicフィルタ
  - [x] __URL__
    - `re.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")`
  - [x] __ユーザ名__
    - `re.compile(r"@[a-zA-Z0-9_]{1,15}")`
  - [x] __ハッシュタグ__
  - [x] __日本語を含まない文__
    - `unicodedata.name`が「CJK UNIFIED, HIRAGANA, KATAKANA」のいずれかであれば日本語の文字とする
- 顔文字フィルタ
  - [x] __顔文字検出__
    1. 正規表現の "［\W\_a-zA-Z]+" に該当する文字列を顔文字候補 X とする。以下、Xに対して処理を行う。
    2. アルファベットをstrip ("twitter?"，"good！！，"「this"などは顔文字ではない)
    3. "。．.、，,・･…〜~-！？!?" の繰り返しを1文字とみなす
    4. Xに"［\W\_]" が含まれない場合はアルファベットからなる文字列なので顔文字候補から除外
    5. 1~4を行ったのち，Xの長さが3以上のものは顔文字とみなす
    6. 顔文字候補 Xでなくとも，"()"または"（）"で囲まれており，内部に日本語が存在しないものは除外
    7. 6はたとえ日本語であっても"T, o, O, ロ, 口, ﾛ, つ, っ, 灬, ノ, ﾉ, c, C"以外の文字と一回はマッチする必要あり
- その他
  - [x] __トークン数__
    - １発話の単語数が6~29に収まらないものは除外
    - mecab、辞書は`/opt/local/lib/mecab/dic/naist-jdic/sys.dic`
    - BPE token数が128を超えるまでcontextを使用
  - [x] __重複度__
    - Jaccard similarity が 閾値 0.5 を超えたら除外
  - [x] __繰り返し表現__
    - len(set(words)) / len(words) が 0.5を下回ったら除外

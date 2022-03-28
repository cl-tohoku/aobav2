# データセット

## 翻訳

- https://github.com/cl-tohoku/wmt2020-resources

```bash
$ git clone --depth 1 https://github.com/moses-smt/mosesdecoder
```

### 翻訳対象のデータセット

```yaml
ConvAI2
Wizard of Wikipedia
Empathetic Dialogues
Daily Dialog
Blended Skill Talk
```

### 結果の比較

- [SpreadSheet](https://docs.google.com/spreadsheets/d/1dTLrkvn7rd5X4J4SoyXqxK08opgIdCx5rn4fjVHdRWQ/edit#gid=1898375196)
- 比較対象
  - [ ] みらい翻訳
    - 実際の翻訳結果を見ると一番有効かもしれないが、大規模データを翻訳するためには有料版を考える必要がある
  - [x] WMT2020
    - 良い翻訳結果が取得可能だが、クリーニングの必要がある
    - こちらのモデルで誤って翻訳されたものをみらい翻訳で翻訳することで、みらい翻訳 API のアクセス頻度を低くする
  - [ ] Google 翻訳
    - 実用的でない


## 翻訳結果の分析

### アラインメント評価
- see [giza++/README.md](./giza++/README.md)
  - 開発・評価セットでは訓練セットの vcb を使用する

```bash
$ cd giza++
$ bash scripts/setup.sh
```

### GIZA および BackTrans を用いたフィルタリング
- [Wolk+'15 - Noisy-parallel and comparable corpora filtering methodology for the extraction of bi-lingual equivalent data at sentence level](https://arxiv.org/abs/1510.04500)

```bash
$ python evaluate/validate_translation.py \
    -a3 giza++/data/daily/train.A3.final \
    -backtrans work/daily/train.ren \
    -output work/daily/score_train.jsonl
```

- arguments
  - `-a3`: GIZA++ による出力ファイル
  - `-backtrans`: WMT モデル (ja -> en) を使用して取得した英訳データ（one lines） 
  - `-output`: 出力ファイル（同時に `args.output.replace('.jsonl', 'is_filtering')` というフィルタリング対象を記載したファイルも作成される）


## モデルに入力するための形式に変換

### 1. スコアファイルに dialogue\_idx, turn\_idx を紐づける

```bash
$ python evaluate/link_idx_w_score_jsl.py <score file> <idx file>

# each line of <idx file>: '\t'.join([dialogue_idx, turn_idx, text])
```

### 2. スコアファイルから onelines 形式に変換
- dialogue 単位で以下の形式に変換
- `u[idx]['is_filter'] == True` の場合は idx+1 以降の発話は切り捨てる
```
u1, u2
u1 <s> u2, u3
u1 <s> u2 <s> u3, u4
```


```bash
$ python evaluate/convert_score_jsl_to_lines.py <fi_score_w_idx>
```


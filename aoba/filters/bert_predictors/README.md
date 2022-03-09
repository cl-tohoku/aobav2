# JSNLI

## データセット
- [黒橋研 JSNLI](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88)

```bash
$ cd ROOT_REPOSITORY
$ bash scripts/download_jsnli.sh

$ wc -l /work02/SLUD2021/datasets/snli/jsnli_1.1/*
  3916 /work02/SLUD2021/datasets/snli/jsnli_1.1/dev.tsv
533005 /work02/SLUD2021/datasets/snli/jsnli_1.1/train_w_filtering.tsv
548014 /work02/SLUD2021/datasets/snli/jsnli_1.1/train_wo_filtering.tsv
```

## モデル

- [BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforsequence#transformers.BertForSequenceClassification)
- [Fine-tuning a pretrained model](https://huggingface.co/transformers/training.html)
- [Trainer](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer)

### 学習

```bash
# train() を実行するよう設定
$ python aoba/filters/nli_predictor.py --dest /work02/SLUD2021/models/jsnli
```

### 予測

```bash
# predict() を実行するよう設定
$ python aoba/filters/nli_predictor.py
```


### 組み込む場合

```python
from nli_predictor import JsnliPredictor

parser = argparse.ArgumentParser(description='')
parser = JsnliPredictor.add_parser(parser)
args = parser.parse_args()

predictor = JsnliPredictor(args.model_jsnli)

premise = "ワクチン打ったら副作用が辛かった。"
hypothesis = "具体的にどんな副作用がありました？"
result = predictor([[premise, hypothesis]])

# [[{'label': 'entailment', 'score': 0.00027919572312384844}, {'label': 'neutral', 'score': 0.0010284266900271177}, {'label': 'contradiction', 'score': 0.9986923933029175}]]
```


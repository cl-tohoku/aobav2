# parsers.py

## MecabParser

```py
from parsers import MecabParser

parser = MecabParser()
text = "吾輩は猫である。名前はまだない。"
parsed_text = parser(text)
```

# normalizers.py

## SentenceNormalizer

```py
from normalizers import SentenceNormalizer

normalizer = SentenceNormalizer()
text = "ﾔｯﾎｰ...（笑）笑www😄..."
normalized_text = normalizer(text)
```

# bert_predictors

## NextUtterancePredictor

```py
from bert_predictors import NextUtterancePredictor

model = "/work01/slud_livechat_2020/mlm-checkpoint-43000-pytorch-model.bin"
contexts = ["今日はいい天気ですね", "外に遊びにいきましょう"]
response_candidates = ["いいですね", "どこいきますか？"]

predictor = NextUtterancePredictor(model)
results = predictor(contexts, response_candidates)
```

## JsnliPredictor

```py
from jsnli import JsnliPredictor

parser = argparse.ArgumentParser(description='')
parser = JsnliPredictor.add_parser(parser)
args = parser.parse_args()

predictor = JsnliPredictor(args.model_jsnli)

premise = "ワクチン打ったら副作用が辛かった。"
hypothesis = "具体的にどんな副作用がありました？"
result = predictor(premise, hypothesis)

# [[{'label': 'entailment', 'score': 0.00027919572312384844}, {'label': 'neutral', 'score': 0.0010284266900271177}, {'label': 'contradiction', 'score': 0.9986923933029175}]]
```
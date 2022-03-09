import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

from glob import glob
from cytoolz import curry

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from torch.utils.data import Dataset

from transformers import BertConfig, TextClassificationPipeline, BertForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

DDIR="/work02/SLUD2021/datasets/snli/jsnli_1.1"
DEFAULT_MODEL = "/work02/SLUD2021/github/src/submodules/jsnli/outputs/best-24000"
Premise = str
Hypothesis = str


class JsnliProcessor(object):
    @classmethod
    def add_parser(cls, parser):
        _jsnli = parser.add_argument_group('Group of JSNLI Datasets')
        _jsnli.add_argument('--fi_train', default=f'{DDIR}/train_w_filtering.tsv', type=str, help='')
        _jsnli.add_argument('--fi_valid', default=f'{DDIR}/dev.tsv', type=str, help='')
        return parser

    def __init__(self, fi_tsv=None):
        self.fi_tsv = fi_tsv
        self.columns = ['label', 'premise', 'hypothesis']
        self.df = self.load(fi_tsv)
        self.tokenizer = self.set_tokenizer()
        self.encoded = self.encode(self.df['text'].values.tolist())

    def set_tokenizer(self):
        from transformers import BertJapaneseTokenizer
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        return tokenizer
        
    def load(self, fi_tsv):
        def concat(premise, hypothesis):
            tmp = f'{premise} [SEP] {hypothesis}'
            return ''.join(tmp.split())
        df = pd.read_csv(open(fi_tsv), sep='\t', header=None)
        df.columns = self.columns
        df['text'] = df.apply(lambda x: concat(x['premise'],  x['hypothesis']), axis=1)
        return df

    def encode(self, texts: List[str]):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')


class JsnliDataset(Dataset, JsnliProcessor):
    def __init__(self, fi_tsv):
        super().__init__(fi_tsv)
        self.labels = {'entailment':0, 'neutral':1, 'contradiction':2}
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k,v in self.encoded.items()}
        item['labels'] = torch.tensor(self.labels[self.df['label'].iloc[idx]])
        return item
    
    def __len__(self,):
        return self.df.shape[0]


class JsnliValidator(object):

    def __init__(self, model):
        self.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self.config = BertConfig.from_pretrained(model, num_labels=3, id2label=self.id2label)
        self.model = BertForSequenceClassification.from_pretrained(model, config=self.config)
        self.tokenizer = self.set_tokenizer()
        # self.classifier = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True)

    def set_tokenizer(self):
        from transformers import BertJapaneseTokenizer
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        return tokenizer

    def calc_score(self, premise:str, hypothesis:str) -> List[dict]:
        input_text = premise + '[SEP]' + hypothesis
        return self.classifier(input_text)

    def validate(self, dataset):
        trainer = Trainer(
            model = self.model,
            tokenizer = self.tokenizer,
            compute_metrics = self.compute_metrics,
            eval_dataset = dataset,
        )
        result = trainer.evaluate()
        return result

    @curry
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


class JsnliPredictor(object):
    @classmethod
    def add_parser(cls, parser):
        _jsnli = parser.add_argument_group('Group of JsnliPredictor')
        _jsnli.add_argument('--model_jsnli', default=DEFAULT_MODEL, type=str, help='')
        return parser

    def __init__(self, model=None, device=0):
        if model is None:
            model = DEFAULT_MODEL
        self.model = BertForSequenceClassification.from_pretrained(model)
        self.tokenizer = self.set_tokenizer()
        self.classifier = TextClassificationPipeline(
            model=self.model, 
            tokenizer=self.tokenizer, 
            return_all_scores=True,
            device=device,
        )

    def set_tokenizer(self):
        from transformers import BertJapaneseTokenizer
        tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        return tokenizer

    def __call__(self, pairs:List[Tuple[Premise, Hypothesis]]) -> List[dict]:
        inputs = ['{} [SEP] {}'.format(premise, hypothesis) for premise, hypothesis in pairs]
        return self.classifier(inputs)


def load_dataset():
    """ bash
    python $0 \
        # --fi_train $FI_TRAIN 
        # --fi_valid $FI_VALID
    """
    parser = argparse.ArgumentParser(description='')
    parser = JsnliProcessor.add_parser(parser)
    args = parser.parse_args()

    valid_dataset = JsnliDataset(args.fi_valid)
    print(valid_dataset[0])


def train():
    """ bash
    python $0 \
        --dest $DIR_OUT \
        --fi_train $FI_TRAIN \
        --fi_valid $FI_VALID
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dest', default='outputs', type=str, help='')
    parser = JsnliProcessor.add_parser(parser)
    args = parser.parse_args()

    config = BertConfig.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    model = BertForSequenceClassification.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        config=config,
    )

    train_dataset = JsnliDataset(args.fi_train)
    valid_dataset = JsnliDataset(args.fi_valid)
    

    training_args = TrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=100,
        output_dir=args.dest,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    trainer.train()
    # trainer.evaluate()


def validate():
    """ bash
    python $0 \
        # --fi_train $FI_TRAIN \
        # --fi_valid $FI_VALID
    """
    parser = argparse.ArgumentParser(description='')
    parser = JsnliProcessor.add_parser(parser)
    args = parser.parse_args()
    
    valid_dataset = JsnliDataset(args.fi_valid)
    
    results = []
    with open('valid_score.jsonl', 'w') as fo:
        for model in glob(DEFAULT_MODEL):
            validator = JsnliValidator(model)
            result = validator.validate(valid_dataset)
            result['model'] = os.path.basename(model).replace('checkpoint-', '')
            results.append(result)
            fo.write(json.dumps(result) + '\n')
        print(f'| WRITE ... {fo.name}')

    df = pd.DataFrame(results).set_index('model').sort_index().round(2)
    cols = ['eval_f1', 'eval_recall', 'eval_precision', 'eval_accuracy']
    df = df[cols].rename({k:k.replace('eval_', '') for k in cols})
    df.to_csv(open('valid_score.tsv', 'w'), sep='\t')
    print(f'| WRITE ... valid_score.tsv')


def predict():
    """ bash
    python $0 \
        # --model_jsnli DEFAULT_MODEL
    """
    parser = argparse.ArgumentParser(description='')
    parser = JsnliPredictor.add_parser(parser)
    args = parser.parse_args()
    
    predictor = JsnliPredictor(args.model_jsnli)

    premise = "ワクチン打ったら副作用が辛かった。"
    # hypothesis = "具体的にどんな副作用がありました？"
    hypothesis = "サッカーしようよ"
    pair = (premise, hypothesis)
    result = predictor([pair])

    print(f'| premise    ... {premise}')
    print(f'| hypothesis ... {hypothesis}')
    print(result)



if __name__ == '__main__':
    predict()
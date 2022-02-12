import os
import sys
import json
import logging
import argparse
from tqdm import tqdm
from pprint import pprint
from difflib import SequenceMatcher
import datasets
from transformers import BertTokenizer

logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.WARNING,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


class BacktransMetrics:
    def __init__(self):
        # self.metric_bleu = datasets.load_metric("bleu")
        self.metric_meteor = datasets.load_metric("meteor")
        # self.metric_rouge = datasets.load_metric("rouge")
        # self.metric_wer = datasets.load_metric("wer")
        self.metric_bertscore = datasets.load_metric("bertscore")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, source:str, target:str):
        # source = " ".join(self.tokenizer.tokenize(source))
        # target = " ".join(self.tokenizer.tokenize(target))
        # bleu_score = self.bleu(source, target)
        meteor_score = self.meteor(source, target)
        # rouge_score = self.rouge(source, target)
        # wer_score = self.wer(source, target)
        bert_score = self.bertscore(source, target)
        return {
            # "BLEU": bleu_score["bleu"],
            "METEOR": meteor_score["meteor"],
            # "WER": wer_score,
            "BERTScore": bert_score["f1"][0],
        }

    def bleu(self, source:str, target:str):
        references = [[source.split()]]
        predictions = [target.split()]
        self.metric_bleu.add_batch(predictions=predictions, references=references)
        return self.metric_bleu.compute()
    
    def meteor(self, source:str, target:str):
        references = [source]
        predictions = [target]
        self.metric_meteor.add_batch(predictions=predictions, references=references)
        return self.metric_meteor.compute()
    
    def rouge(self, source:str, target:str):
        references = source.split()
        predictions = target.split()
        self.metric_rouge.add_batch(predictions=predictions, references=references)
        return self.metric_rouge.compute()
    
    def wer(self, source:str, target:str):
        references = [source]
        predictions = [target]
        self.metric_wer.add_batch(predictions=predictions, references=references)
        return self.metric_wer.compute()
    
    def bertscore(self, source:str, target:str):
        references = [source]
        predictions = [target]
        self.metric_bertscore.add_batch(predictions=predictions, references=references)
        # return self.metric_bertscore.compute(predictions=predictions, references=references, lang="en")
        return self.metric_bertscore.compute(model_type="bert-base-uncased")


def eval_backtrans(args):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fi_original", default="", help="Original File (EN)")
    parser.add_argument("--fi_backtrans", default="", help="Re-Translated File (EN)")
    parser.add_argument("--fo_score", default="backtrans.convai2.context.jsonl", help="Re-Translated File (EN)")
    args = parser.parse_args()
    
    metrics = BacktransMetrics()
    with open(args.fo_score, "w") as fo:
        for txt_org, txt_back in tqdm(zip(open(args.fi_original), open(args.fi_backtrans))):
            result = metrics(txt_org.strip(), txt_back.strip())
            fo.write(json.dumps(result) + "\n")
        print(f"| WRITE ... {fo.name}")



if __name__ == "__main__":
    source = "she was interested in world history because she read the book"
    target = "she read the book because she was interested in world history"
    metrics = BacktransMetrics()
    result = metrics(source, target)
    pprint(result)
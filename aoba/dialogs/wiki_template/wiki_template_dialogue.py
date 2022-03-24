import argparse
import dataclasses
from collections import defaultdict
import gzip
from itertools import groupby
import json
import logging
import random
import re
from os.path import dirname, join
import sys
from typing import List

from tqdm import tqdm

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

sys.path.append(join(dirname(__file__), "../../filters"))
from parsers import MecabParser


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


ROOT_REPOSITORY = join(dirname(__file__), "../"*3)


@dataclasses.dataclass
class Knowledge:
    subject: str
    object: str
    relation: str
    def __getitem__(self, key:str):
        return getattr(self, key)


class CleanUpText():
    def erase_link(self, matchobj):
        """ clean_up_textのサブ関数。実際に記事中に表記されている文字列だけを抽出する """
        # matchobj.group('link') = 実際の表記, matchobj.group('linked') = 多分リンク飛んだ先のURL
        return matchobj.group('link')

    def replace(self, text):
        text = text.rstrip('である') + 'みたいですよ。'
        text = text.replace('-', '')
        text = text.replace('である', 'らしい')
        text = text.replace('ないが', 'ないらしいんですが')
        text = text.replace('であったが', 'だったそうですが')
        return text

    def clean_up_detail_sent(self, knowledge):
        """ 理由文を整形した自分自身を返す """
        # 間の空白削除
        raw_detail_sent = knowledge['text']
        raw_text = ''.join(raw_detail_sent.strip().split())
        # <sentence>~<eos>の~の部分をとる
        # Wikidataの場合、<sentence>~<eos>タグがないのでNoneになる
        if re.search(r'(?<=<sentence>).*?(?=<eos>)', raw_text) is not None:
            matchobj = re.search(r'(?<=<sentence>).*?(?=<eos>)', raw_text)
            text = matchobj.group()
        else:
            text = raw_text
        # リンク関係のいらない文字列を整形
        text = re.sub(r'/\((?P<link>.*?)/(.*?)\)/', self.erase_link, text)
        text = re.sub(r'_\(.*?\)', '', text)
        text = re.sub(r'（.*）', '', text)
        text = re.sub(r'\(.*\)', '', text)
        text = self.replace(text)
        target = knowledge['subject']
        text = text.replace(f'{target}は、', '')
        return text


class WikipediaTemplateDialogue(object):
    def __init__(self):
        self.path = join(dirname(__file__), "datasets.yml")
        self.cfg = OmegaConf.load(self.path)
        self.triple = self.load_triple(self.cfg["wiki_knowledge"])
        self.response_templates = [json.loads(line) for line in open(join(ROOT_REPOSITORY, self.cfg["wiki_response_template"]))]
        self.context_templates = json.load(open(join(ROOT_REPOSITORY, self.cfg["wiki_context_template"])))
        self.tagger = MecabParser()
        self.cleaner = CleanUpText()

    def load_triple(self, fi_triple):
        outputs = defaultdict(list)
        for line in tqdm(gzip.open(fi_triple, 'rt'), desc='[LOAD] triple'):
            # line = {'subject':str, 'object':str, 'relation':str}
            line = json.loads(line.strip())
            outputs[line['subject']].append(line)
        return outputs

    @property
    def deep_phrase(self):
        return [d['question_phrase'] for d in self.response_templates]

    @property
    def shallow_phrase(self):
        return ('知って', '教えて', 'ご存知', '詳しい？')

    @property
    def shallow_context(self):
        return [
            '{subj}って知ってる？',
            '{subj}について知っていますか？',
            '{subj}について教えて',
            '{subj}はご存知ですか？',
            '{subj}について詳しいですか？',
            '{subj}って聞いたことありますか？'
        ]

    def is_shallow(self, context):
        """ 「〜って知ってる？」のような特定の知識（場所など）を尋ねない質問 """
        return any(x in context for x in self.shallow_phrase)

    def extract_noun_phrases(self, text):
        """ 名詞句を抽出 """
        try:
            parsed_text = self.tagger(text)
            return [
                ''.join([n.surface for n in noun_phrase]) \
                for k, noun_phrase in groupby(parsed_text, key=lambda x: x.pos1=='名詞') \
                if k
            ]
        except TypeError:
            logger.warning("\033[31m" + f"TypeError: WikipediaTemplateDialogue.extract_noun_phrases: {text}" + "\033[0m")
            return []

    def conditioned_knowledge(self, knowledges, template_candidates):
        """ relation が一致するペアを返す。例えば、場所としての「東京」か、観光地としての「東京」か。"""
        pairs = []
        for template in template_candidates:
            for knowledge in knowledges:
                if knowledge['relation'] == template['relation']:
                    pairs.append((knowledge, template))
        if pairs:
            # 複数知識が取得された場合はランダムに一つ選択
            knowledge, template = random.choice(pairs)
            # contexts = self.context_templates.get(knowledge['relation'], []) + self.shallow_context
            # context = random.choice(contexts).format(subj=knowledge['subject'])
            response = template['template'].format(subj=knowledge['subject'], obj=knowledge['object'])
            return knowledge, response

    def clean_knowledge(self, knowledge):
        return self.cleaner.clean_up_detail_sent(knowledge)

    @staticmethod
    def jaccard(a, b):
        return len(set(a) & set(b)) / len(set(a) | set(b))

    def join_response(self, response, knowledge):
        """ テンプレートと知識の重複がない場合は連結 """
        iou = self.jaccard(
            [e.surface for e in self.tagger(response)],
            [e.surface for e in self.tagger(knowledge)]
        )
        return response if iou > 0.4 else response + knowledge

    def __call__(self, context):
        # 名詞句抽出
        noun_phrases = self.extract_noun_phrases(context)
        if len(noun_phrases) > 0:
            # 対象の名詞句を最後尾の名詞句とする（経験的に）
            query = noun_phrases[-1]
            # クエリに対応する知識を抽出
            knowledges: List[dict] = self.triple.get(query)
            if knowledges is not None:
                # 特定の知識を尋ねない/尋ねる質問で分岐
                if self.is_shallow(context):
                    template_candidates = self.response_templates
                else:
                    # 質問フレーズ（「どこ」「何」）を特定し、対応するテンプレートを取得
                    template_candidates = filter(lambda x: x['question_phrase'] in context, self.response_templates)
                # 質問の対象としている関係（どこ → 場所）に対応する三つ組を返す
                outputs = self.conditioned_knowledge(knowledges, template_candidates)
                if outputs is not None:
                    knowledge, response = outputs
                    # 整形
                    cleaned_knowledge = self.clean_knowledge(knowledge)
                    response = self.join_response(response, cleaned_knowledge)
                    return knowledge, context, response


def interactive():
    template_dialogue = WikipediaTemplateDialogue()
    while True:
        try:
            context = input('Input > ')
            output = template_dialogue(context)
            if output is None:
                print('Not Found')
                continue
            knowledge, response = output
            knowledge = template_dialogue.clean_knowledge(knowledge['text'])
            print(response)
            print(knowledge)
        except KeyboardInterrupt:
            sys.exit()


def craete_doyouknow_dialog():
    """ 名詞リストから `{名詞}って知ってますか？` を context とする対話対を作成 """ 
    template_dialogue = WikipediaTemplateDialogue()
    with open('tmp/doyouknow.context', 'w') as fc, open('tmp/doyouknow.response', 'w') as fr:
        for line in tqdm(open('/work02/SLUD2021/datasets/wikipedia/pageview_most_common_ver2.csv'), desc='[LOAD] nouns'):
            line = line.strip().split()
            if len(line) != 2:
                continue
            else:
                title, pv = line
                if int(pv) < 10:
                    continue
                context = f'{title}って知ってますか？'
                output = template_dialogue(context)
                if output is None:
                    continue
                _, context, response = output
                print(context, file=fc)
                print(response, file=fr)



if __name__ == "__main__":
    # interactive()
    template_dialogue = WikipediaTemplateDialogue()
    context = "東京タワーって知ってますか？"
    response = template_dialogue(context)
    if response:
        knowledge, response = response
        print(context)
        print(response)

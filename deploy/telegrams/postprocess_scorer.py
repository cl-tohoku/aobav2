from typing import List, Dict, Any

import MeCab

from . import SpmTokenizer

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from submodules.jsnli import JsnliPredictor


class PostprocessScorer:
    def __init__(self, args):
        self.args = args
        self.inf_dct = {}
        self.sif_param = self.args.sif_param
        self.add_inf_dct(self.args.init_wordfreq_fname)
        self.tokenizer = SpmTokenizer(self.args.spm)
        self.mecab = MeCab.Tagger("-Owakati")
        self.jsnli_predictor = JsnliPredictor(self.args.jsnli_model_fname, device=0)

    def __call__(self, re_ranking_results: List[Dict[str, Any]]):
        """
        Add informativeness score and jaccard score to re_ranking_results
        """
        return [self.add_post_scores(result_dct) for result_dct in re_ranking_results]

    def add_post_scores(self, result_dct):
        # tokenize
        response_tokens = self.tokenize(result_dct['response'])
        context_tokens_lis = [self.tokenize(context) for context in result_dct['contexts']]
        full_context_tokens_lis = [self.tokenize(context) for context in result_dct['full_contexts']]
        # compute jaccard score
        result_dct['jaccard'] = self.compute_tokens_jaccard(response_tokens, context_tokens_lis)
        result_dct['jaccard_usr'] = self.compute_tokens_jaccard(response_tokens, context_tokens_lis, usrside=True)
        result_dct['jaccard_mix'] = self.compute_tokens_jaccard(response_tokens, full_context_tokens_lis, mix=True)
        result_dct['highest_jaccard'] = max(result_dct['jaccard'])
        result_dct['highest_jaccard_usr'] = max(result_dct['jaccard_usr'])
        result_dct['highest_jaccard_mix'] = max(result_dct['jaccard_mix'])
        # compute SIF score
        result_dct['inf'] = self.get_inf_tokens_score(response_tokens)
        # compute duplicate score (The more duplicate words the sentence has, the smaller this score)
        result_dct['duplicate_score'] = self.compute_duplicate_score(result_dct['response'])
        # compute JSNLI score
        result_dct['jsnli_label'], result_dct['jsnli_score'] = self.compute_jsnli_score(result_dct['full_contexts'][-1], result_dct['response'])

        result_dct['prev_jsnli_label'], result_dct['prev_jsnli_score'] = self.compute_jsnli_score(result_dct['full_contexts'][-2], result_dct['response'])
        # compute wiki score
        # result_dct['wiki_score'] = self.compute_wiki_score(result_dct['full_contexts'][-1], result_dct['response'])

        return result_dct

    def tokenize(self, sentence):
        return self.tokenizer.encode(sentence).split()

    def add_inf_dct(self, wordfreq_fname):
        """
        Add word frequency information to self.dct
        The format of 'wordfreq_fname' is following:
            <token_1> <number of apperances in a corpus>
            <token_2> <number of apperances in a corpus>
                :
        """
        if not wordfreq_fname:
            return
        total_num_word = 0
        with open(wordfreq_fname) as wordfreq_f:
            for l in wordfreq_f:
                _, word_count = l.strip().split(' ')
                total_num_word += float(word_count)
            wordfreq_f.seek(0)
            for l in wordfreq_f:
                word, word_count = l.strip().split(' ')
                self.inf_dct[word] = self.sif_param / (self.sif_param + float(word_count) / total_num_word)

    def token2infscore(self, word: str) -> float:
        return self.inf_dct[word] if word in self.inf_dct else 1.0

    def get_inf_of_tokens(self, words: List[str]) -> List[float]:
        return [self.token2infscore(word) for word in words]

    def get_inf_tokens_score(self, tokens):
        """
        Args:
            words: List of tokens
        Return:
            List of informativeness score
        """
        return sum(self.get_inf_of_tokens(tokens))

    def compute_jaccard_sim(self, set_a, set_b):
        return len(set_a & set_b) / len(set_a | set_b)

    def compute_tokens_jaccard(self, tokens_a, tokens_b_lis, usrside=False, mix=False):
        tokens_a = set(tokens_a)
        tokens_b_lis = [set(tokens_b) for tokens_b in tokens_b_lis]
        if mix:
            compute_flg = [True] * len(tokens_b_lis)
        else:
            compute_flg = [True if i % 2 == 1 else False for i in range(len(tokens_b_lis))][::-1]
            compute_flg = [not flg for flg in compute_flg] if usrside else compute_flg
        return [self.compute_jaccard_sim(tokens_a, tokens_b_lis[i]) if compute_flg[i] else 0.
                for i in range(len(tokens_b_lis))]

    def compute_duplicate_score(self, text: str) -> float:
        """compute duplicate score (The more duplicate words the sentence has, the smaller this score)"""
        tokens = self.mecab.parse(text)
        duplicate_score = len(set(tokens)) / len(tokens)

        return duplicate_score

    # add filtering by jsnli label (2021/10/05 kishinami)
    def compute_jsnli_score(self, context, response):
        result = self.jsnli_predictor([(context, response)])[0]
        result = {d['label']:d['score'] for d in result}
        jsnli_label = max(result, key=result.get)
        jsnli_score = result['contradiction']
        return jsnli_label, jsnli_score

    # add filtering by wiki score (2021/10/11 miyawaki)
    def compute_wiki_score(self, context, response):
        pass

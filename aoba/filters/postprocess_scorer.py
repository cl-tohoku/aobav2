from os.path import dirname, isfile
from typing import List, Dict, Any
import sys

from overrides import overrides

sys.path.append(dirname(__file__))
from parsers import MecabParser


class PostprocessScorer:
    def __init__(self, fi_wordfreq, _jaccard=True, _sif=True, _duplicate=True):
        self.token2sif = self.load_sif(fi_wordfreq) if isfile(fi_wordfreq) else dict()
        self.tagger = MecabParser()
        self._jaccard = _jaccard
        self._sif = _sif
        self._duplicate = _duplicate

    def load_sif(self, fi_wordfreq, sif_param=1e-3) -> Dict[str, float]:
        """ 単語の出現頻度情報を辞書形式でロード
        The format of "wordfreq_fname" is following:
            <token_1> <number of apperances in a corpus>
                :
        """
        counter = dict()
        for line in open(fi_wordfreq):
            token, cnt = line.strip().split(" ")
            counter[token] = float(cnt)
        n_total = sum(counter.values())
        sif = lambda x: sif_param / (sif_param + x/n_total)
        return {token: sif(cnt) for token, cnt in counter.items()}

    def wakati(self, text):
        return [token.surface for token in self.tagger(text)] 

    def __call__(self, result: Dict[str, Any]):
        """ Add informativeness score and jaccard score to re_ranking_results """
        # tokenize
        response: List[str] = self.wakati(result["response"])
        contexts: List[List[str]] = [self.wakati(context) for context in result["full_contexts"]]
        # compute jaccard score
        if self._jaccard:
            result["jaccard_mix"] = self.calc_jaccards(response, contexts, is_mix=True, is_userside=False)
            result["highest_jaccard_mix"] = max(result["jaccard_mix"])
        # compute SIF score
        if self._sif:
            result["inf"] = self.calc_sif(response)
        # compute duplicate score (The more duplicate words the sentence has, the smaller this score)
        if self._duplicate:
            result["duplicate_score"] = self.calc_duplicates(response)
        return result

    @staticmethod
    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    def calc_jaccards(self, response, contexts:List[List[str]], is_userside=False, is_mix=False):
        response = set(response)
        contexts = [set(context) for context in contexts]
        if is_mix:
            compute_flg = [True] * len(contexts)
        else:
            compute_flg = [i%2==1 for i in range(len(contexts))][::-1]
            if is_userside:
                compute_flg = [not flg for flg in compute_flg]
        jaccards = []
        for i, context in enumerate(contexts):
            jaccards.append(self.jaccard(response, context) * int(compute_flg[i]))
        return jaccards

    def calc_sif(self, tokens:List[str]) -> float:
        return sum(self.token2sif.get(token, 1.0) for token in tokens)

    def calc_duplicates(self, tokens:List[str]) -> float:
        return len(set(tokens)) / len(tokens)
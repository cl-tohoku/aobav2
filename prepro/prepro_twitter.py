import argparse
import gzip
import multiprocessing as mp
from pathlib import Path
from typing import Generator

from tqdm import tqdm
from omegaconf import OmegaConf

ROOT_REPOSITORY = Path(__file__).parents[1]
from aoba import (
    DialogFilter,
    PostprocessScorer,
    MecabParser,
    SentenceNormalizer,
    NextUtterancePredictor,
    JsnliPredictor
)


parser = argparse.ArgumentParser(description="Twitter データの前処理")
parser.add_argument("--ddir", type=str, help="path to input context file")
parser.add_argument("--basename", type=str, help="path to input response file")
parser.add_argument("--dest", type=str, default=None, help="path to output context file")
parser.add_argument("--fi_config", type=str, default=f"{ROOT_REPOSITORY}/params_prepro.yml", help="path to output response file")

args = parser.parse_args()
for key, value in OmegaConf.load(args.f_cfg).__dict__.items():
    setattr(args, key, value)

args.fi_context  = Path(args.ddir) / f"{args.basename}.context"
args.fi_response = Path(args.ddir) / f"{args.basename}.response"
dest = args.dest if args.dest is not None else args.ddir
args.fo_context  = Path(dest) / f"filtered_{args.basename}.context"
args.fo_response = Path(dest) / f"filtered_{args.basename}.response"

tagger = MecabParser()
normalizer = SentenceNormalizer()

dialog_filter = DialogFilter(
    args.fi_ng_word,            # ng 単語を含まないか
    args.min_v, args.max_v,     # min_v <= 単語数 <= max_v か
    args.kana_ratio_thrs,       # 仮名の割合が kana_ratio_thrs 以上か
    args._parentheses,          # 括弧を含まないか
)

postprocess_scorer = PostprocessScorer(
    args.wordfreq_file,
    _jaccard = args.mix_jac_thr is not None,            # jaccard の計算をするか
    _sif = args.info_score_thr is not None,             # SIF の計算をするか
    _duplicate = args.duplicate_score_thr is not None,  # 重複度の計算をするか
)

nup_predictor = NextUtterancePredictor(model_file=args.model_nup, device=0) if args._nup else None
jsnli_predictor = JsnliPredictor(args.model_jsnli, device=0) if args._jsnli else None

# ===============


def load(path) -> Generator:
    print("\033[34m" + f"| READ ... {path}" + "\033[0m")
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def filter_fn(pair) -> tuple:
    # normalize
    context = normalizer(pair[0])
    response = normalizer(pair[1])
    # filtering
    if dialog_filter(context) or dialog_filter(response):
        return
    # postprocess scorer
    result = postprocess_scorer({
        "response": tagger(response),
        "full_contexts": [tagger(con) for con in context.split("<s>")],
    })
    if ("highest_jaccard_mix" in result) and (result["highest_jaccard_mix"] > args.mix_jac_thr):
        return
    if ("inf" in result) and (result["inf"] < args.info_score_thr):
        return
    if ("duplicate_score" in result) and (result["duplicate_score"] <= args.duplicate_score_thr):
        return
    # jsnli
    if jsnli_predictor is not None:
        result = {res['label']:res['score'] for res in jsnli_predictor([[context.split('<s>')[-1], response]])}
        if max(result, key=result.get) == 'contradiction':
            return
    # nup
    if nup_predictor is not None:
        result = nup_predictor(context.split('<s>').strip(), [response])[0][0]
        if result < args.next_uttr_thr:
            return
    return context, response


buff = []
with open(args.fo_context, "w") as fc, open(args.fo_response, "w") as fr:
    for context, response in tqdm(zip(load(args.fi_context), load(args.fi_response))):
        buff.append((context, response))
        if len(buff) >= 1024:
            with mp.Pool(processes=args.n_worker) as pool:
                for idx, pair in enumerate(pool.map(filter_fn, buff)):
                    if pair is not None:
                        print(pair[0], file=fc)
                        print(pair[1], file=fr)
            buff = []

    print("\033[34m" + f"| WRITE ... {fc.name}" + "\033[0m")
    print("\033[34m" + f"| WRITE ... {fr.name}" + "\033[0m")
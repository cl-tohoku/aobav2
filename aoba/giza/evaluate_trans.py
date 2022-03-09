import argparse
import logging
from os.path import dirname
import sys
import re

import pandas as pd

sys.path.append(dirname(__file__))
# from sbert import StsEncoder
from backtrans import BacktransMetrics


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.WARNING,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


class TransEvaluator(BacktransMetrics):
    def __init__(self):
        super().__init__()

    @classmethod
    def load_a3(cls, fi_a3) -> pd.DataFrame:
        results = []
        for line in open(fi_a3):
            if line.startswith('# Sentence pair'):
                pattern = 'source length ([0-9]+?) target length ([0-9]+?) alignment'
                m = re.compile(pattern).search(line)
                len_src = int(m.groups()[0])
                len_tgt = int(m.groups()[1])
                score = float(line.split(':')[-1])
                nrm_score = score * (10**len_tgt)
            elif line.startswith('NULL'):
                txt_src = ' '.join(re.sub('\(.*?\)', '', line).split()[1:])
                results.append({
                    'ln_en': len_src,
                    'ln_ja': len_tgt,
                    'txt_en': txt_src,
                    'txt_ja': txt_tgt,
                    'aln_score': score,
                    'aln_nrm_score': nrm_score,
                })
            else:
                txt_tgt = line.strip()
        return pd.DataFrame(results)


def clip_outlier(series, bias=1.5):
    """ 外れ値のクリッピング """
    q1 = series.quantile(.25)
    q3 = series.quantile(.75)
    iqr = q3 - q1
    outlier_min = q1 - (iqr) * bias
    outlier_max = q3 + (iqr) * bias
    return outlier_min, outlier_max


def save_score_file(df, output_file):
    with open(output_file, "w") as fo_jsl:
        df.to_json(fo_jsl, orient='records', force_ascii=False, lines=True)
        logger.info(f'| WRITE ... {fo_jsl.name}')


if __name__ == '__main__':
    """ bash
    $ python $0 \
        -a3 {a3_file} \
        -backtrans {backtrans_file} \
        -output {output_file}
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a3', '--a3_file', type=str, help='alignment file from GIZA++')
    parser.add_argument('-backtrans', '--backtrans_file', type=str, help='re-translated file in English (one lines)')
    parser.add_argument('-score', '--score_file', default=None, type=str, help='scored jsonl file from this file')
    parser.add_argument('-output', '--output_file', default='output.jsonl', type=str)
    args = parser.parse_args()
    
    # スコアファイルが保存されている場合はロードして使用
    if args.score_file is not None:
        df = pd.read_json(args.score_file, orient='records', lines=True)

    else:
        evaluator = TransEvaluator()
        df_align = TransEvaluator.load_a3(args.a3_file)
        df_btrans = pd.DataFrame([line.strip() for line in open(args.backtrans_file)], columns=["txt_ren"])
        assert df_align.shape[0] == df_btrans.shape[0], f'SizeMismatchError: {args.a3_file} and {args.backtrans}'
        df = pd.concat([df_align, df_btrans], axis=1)

        df['ln_ren'] = df.apply(lambda x: len(x['txt_ren'].split()), axis=1)
        df['ln_ja-en'] = df.apply(lambda x: x['ln_ja']-x['ln_en'], axis=1)
        df['ln_ren-en'] = df.apply(lambda x: x['ln_ren']-x['ln_en'], axis=1)

        _min, _max = clip_outlier(df['aln_nrm_score'])
        df['aln_outlier'] = (df['aln_nrm_score'] < _min) | (_max < df['aln_nrm_score'])
        vmin_aln, vmax_aln = df[~df['aln_outlier']]['aln_nrm_score'].quantile([0.025, 0.975]).values

        # df['sim_score'] = df.apply(lambda x: float(sts_encoder(x['txt_en'], x['txt_ren']).item()), axis=1)
        df['meteor_score'] = df.apply(lambda x: evaluator.meteor(x['txt_en'], x['txt_ren'])['meteor'], axis=1)
        df['bert_score'] = df.apply(lambda x: evaluator.bertscore(x['txt_en'], x['txt_ren'])['f1'][0], axis=1)
        vmin_sim = df['sim_score'].quantile(0.05)
        vmin_meteor = df['meteor_score'].quantile(0.05)
        vmin_bertscore = df['bert_score'].quantile(0.05)

        df['is_filter'] = df.apply(lambda x: 
            x['aln_outlier'] \
            or (x['aln_nrm_score'] <= vmin_aln) \
            or (vmax_aln <= x['aln_nrm_score']) \
            or (x['sim_score'] <= vmin_sim) \
            or (x['meteor_score'] <= vmin_meteor) \
            or (x['bert_score'] <= vmin_bertscore) \
        , axis=1)
        
        save_score_file(df, args.output_file)
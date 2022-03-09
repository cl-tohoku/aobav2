import os
import sys
import json
from tqdm import tqdm
import argparse
from pprint import pprint

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from parsers import MecabParser



def create_arg_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fi_con', default='', type=str, help='')
    parser.add_argument('--fi_res', default='', type=str, help='')
    return parser


def run():
    parser = create_arg_parser()
    args = parser.parse_args()

    tagger = MecabParser()

    fi_wiki="/work02/SLUD2021/datasets/wikipedia/formated_wikidumps/formated_wikidump_threshold_19.jsonl"
    titles = []
    for d in open(fi_wiki):
        title = json.loads(d)['title']
        try:
            titles.extend([x.surface for x in tagger([title])[0]])
        except:
            titles.append(title)
    titles = set(titles)
    print(f'# of titles ... {len(titles)}')

    fo_con = args.fi_con.replace('.context', '_w_ne.context')
    fo_res = args.fi_res.replace('.response', '_w_ne.response')

    cnt = 0

    with open(fo_con, 'w') as fc, open(fo_res, 'w') as fr:
        for con, res in tqdm(zip(open(args.fi_con), open(args.fi_res))):
            _res = tagger([res])[0]
            # if any(x.pos2 == '固有名詞' for x in _res):
            for noun in filter(lambda x: x.pos2 == '固有名詞', _res):
                # for title in titles:
                #     if title in ''.join(res.split()):
                if noun.surface in titles:
                    print(con.strip(), file=fc)
                    print(res.strip(), file=fr)
                    cnt += 1
                    break
        print(f'| WRITE ... {fc.name}')
        print(f'| WRITE ... {fr.name}')
        print(f'| extracted ... {cnt}')



if __name__ == '__main__':
    run()

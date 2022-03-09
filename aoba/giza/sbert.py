""" SentenceTransformer による類似度評価 """
import argparse

from sentence_transformers import SentenceTransformer, util


class StsEncoder(object):
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-mpnet-base-v2')

    def __call__(self, text1:str, text2:str):
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        sim = util.cos_sim(emb1, emb2)
        return sim


def eval_similarity():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--orig', type=str)
    parser.add_argument('--back', type=str)
    parser.add_argument('--fo_sim', default='sbert_sim.py', type=str)
    args = parser.parse_args()
    encoder = StsEncoder()

    with open(args.fo_sim, 'w') as fo:
        for orig, back in zip(open(args.orig), open(args.back)):
            orig = orig.strip()
            back = back.strip()
            sim = encoder(orig, back)
            print(sim, file=fo)
        print(f'| WRITE ... {fo.name}')



if __name__ == '__main__':
    source = "she was interested in world history because she read the book"
    target = "she read the book because she was interested in world history"
    encoder = StsEncoder()
    result = encoder(source, target)
    print(result)

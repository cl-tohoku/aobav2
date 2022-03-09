# aobav2/prepro に移動

import gzip
import logging
import multiprocessing as mp
import os
from os.path import dirname, join
import sys

from omegaconf import OmegaConf

DIR_FID = join(dirname(__file__), "../")
sys.path.append(join(DIR_FID, "../"))

from wiki_template.wiki_template_dialogue import WikipediaTemplateDialogue


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)



fo = "data/doyouknow.txt"
os.makedirs(dirname(fo), exist_ok=True)

template_dialogue = WikipediaTemplateDialogue()
# response = template_dialogue("東京タワーって知ってる？")


cfg = OmegaConf.load(join(DIR_FID, "datasets.yml"))
wiki_pv = filter(lambda x: (len(x)==2) and (int(x[1]) >= 10), [line.split() for line in gzip.open(cfg.wikipedia["pageview"], "rt")])

def search_knowledge(line):
    title, pv = line
    context = f"{title}って知ってますか？"
    return template_dialogue(context)


with mp.Pool(processes=4) as pool, open(fo, "w") as fo_txt:
    for outputs in pool.map(search_knowledge, wiki_pv):
        if outputs is not None:
            knowledge, context, response = outputs
            fo_txt.write(f"{context}\t{response}\n")
    logger.info(f"WRITE ... {fo_txt.name}")



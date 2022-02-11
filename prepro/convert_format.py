import argparse
import importlib
import logging
from pathlib import Path
from typing import Generator
import sys

from tqdm import tqdm
from omegaconf import OmegaConf


ROOT_REPOSITORY = Path(__file__).parents[1]

logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(description="To create future-aware corpus")
    parser.add_argument("data", type=str, help="key of datasets.yml")
    parser.add_argument("--dtype", default="train,valid,test", help="datatype")
    parser.add_argument("--all_response", action="store_true")
    parser.add_argument("--dest", type=str, help="output dir")
    parser.add_argument("--sep", default=" <s> ", help="separator")
    return parser


def convert_format(data, all_response=False) -> Generator:
    for dialogue in data:
        for idx in range(1, len(dialogue)):
            if all_response:
                context = dialogue[:idx]
                response = dialogue[idx:]
            else:
                context = dialogue[:idx]
                response = dialogue[idx:idx]

            yield context, response

def main(fi):
    parser = create_parser()
    args = parser.parse_args()

    with open(args.con_output, "w") as con_output_fi, \
        open(args.res_output, "w") as res_output_fi:
        for line in tqdm(fi):
            utterances = line.strip().split(" " + args.sep + " ")
            for i in range(1, len(utterances)):
                print(utterances[:i], file=con_output_fi)
                print(utterances[i:], file=res_output_fi)



if __name__ == "__main__":
    """ bash
    python $0 dailydialog
    """

    parser = create_parser()
    args = parser.parse_args()
    
    cfg_file = ROOT_REPOSITORY / "datasets.yml"
    datasets = OmegaConf.load(cfg_file)
    cfg = datasets[args.data]

    for dtype in args.dtype.split(","):
        path = Path(cfg.path[dtype])
        module = importlib.import_module(cfg["format"])
        data_class = getattr(module, cfg["class"])
        data = data_class(path)
        
        dest = Path(args.dest if args.dest else path.parent)
        fo_context = dest / f"{dtype}.context"
        fo_response = dest / f"{dtype}.response"

        if fo_context.is_file() and fo_response.is_file():
            logger.warning("\033[31m" + "output file is existed! (return)" + "\033[0m")
            logger.warning(f"  - context ... {fo_context}")
            logger.warning(f"  - response ... {fo_response}")
        else:
            with open(fo_context, "w") as fc, open(fo_response, "w") as fr:
                for context, response in convert_format(data, args.all_response):
                    fc.write(args.sep.join(context) + "\n")
                    fr.write(args.sep.join(response) + "\n")
                logger.info(f"WRITE ... {fc.name}")
                logger.info(f"WRITE ... {fr.name}")
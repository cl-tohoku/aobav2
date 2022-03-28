import argparse
import gzip
import importlib
import json
from pathlib import Path
from typing import Generator, List

from logzero import logger
from tqdm import tqdm
from omegaconf import OmegaConf


Episode = Generator # List[str]
ROOT_REPOSITORY = Path(__file__).parents[1]


class Loader:
    def __init__(self, args):
        self.args = args
        self.name = args.name
        self.method = getattr(self, args.name)
        self.sep = args.sep
        _pathes = OmegaConf.load(args.config)[args.name]
        _dtypes = _pathes.keys() if args.dtype is None else args.dtype.split(",")
        self.pathes = {d:_pathes[d] for d in _dtypes}

    def _dest(self, path):
        dest = Path(self.args.dest) or Path(path).parent
        dest.mkdir(exist_ok=True)
        return dest

    def _basename(self, path, suffix=""):
        basename = Path(path).stem
        return basename.split(".")[0] + suffix

    @classmethod
    def add_parser(cls, parser):
        parser.add_argument("name", type=str, help="key of datasets.yml")
        parser.add_argument("--dtype", type=str, help="datatype")
        parser.add_argument("--config", default=f"{ROOT_REPOSITORY}/datasets.yml")
        parser.add_argument("--dest", type=str, help="output dir")
        parser.add_argument("--sep", default=" <s> ", help="separator")
        parser.add_argument("--use-history", action="store_true")
        parser.add_argument("--use-future", action="store_true")
        return parser

    @staticmethod
    def load(path):
        logger.info(f"| LOAD ... {path}")
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rt") as fi:
            if ".jsonl" in path:
                for line in tqdm(fi, desc="[load] "):
                    yield json.loads(line.strip())
            elif ".json" in path:
                for line in tqdm(json.load(fi), desc="[load] "):
                    yield line
            else:
                for line in tqdm(fi, desc="[load] "):
                    yield line.strip()

    def blended_skill_talk(self, path) -> Episode:
        for instance in self.load(path):
            """ instance
            {
                "personas": [
                    ['i m 49 , male and live in dublin , ireland.', 'i have one sister and a niece and nephew.'],
                    ['i work as an electrician.', 'i always sleep 8 hours a day.']
                ], 
                "context_dataset": "wizard_of_wikipedia", 
                "free_turker_utterance": "That sounds dangerous. Is it worth doing such a dangerous job?",
                "guided_turker_utterance": "Wekk it is okay is you are well trained.  There are three levels: Apprentice, journeyman and Master.", 
                "additional_context": "Electrician", 
                "dialog": [
                    [0, 'Which level are you at?'],
                    [1, 'I received on-the-job training when i first started'],
                    ...
                ], 
                "suggestions":, 
                "suggestion_orders":, 
                "chosen_suggestions":, 
                "chosen_suggestion_texts":, 
                "workers":, 
                "bad_workers":, 
                "acceptability_violations":, 
                "hit_ids":, 
                "assignment_ids":, 
                "label_candidates":,
            }
            """
            yield [t[-1] for t in instance["dialog"]]

    def convai2(self, path) -> Episode:
        prev_dix = -1
        episode = []
        for instance in self.load(path):
            """ instance
            {
                "dialogue_idx": 0,
                "turn_idx": "0",
                "self_persona": [
                    "i like to remodel homes.",
                    "i like to go hunting.",
                    "i like to shoot a bow.",
                    "my favorite holiday is halloween."
                ],
                "other_persona": [
                    "i like canning and whittling.",
                    "to stay in shape , i chase cheetahs at the zoo.",
                    "in high school , i came in 6th in the 100 meter dash.",
                    "i eat exclusively meat."
                ],
                "context": "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .",
                "response": "you must be very fast . hunting is one of my favorite hobbies ."
            }
            """
            if prev_dix != instance["dialogue_idx"]:
                yield episode
                episode = []
                prev_dix = instance["dialogue_idx"]
            episode.extend([
                instance["context"], instance["response"]
            ])
        yield episode


    def dailydialog(self, path) -> Episode:
        for instance in self.load(path):
            """ instance
            {
                "fold": "validation",
                "topic": "tourism",
                "dialogue": [
                    {"emotion": "no_emotion", "act": "question",  "text": "Good morning , sir . Is there a bank near here ?"}, 
                    {"emotion": "no_emotion", "act": "inform",    "text": "There is one . 5 blocks away from here ?"},
                ]
            }
            """
            yield [t["text"] for t in instance["dialogue"]]

    def empathetic_dialogues(self, path) -> Episode:
        for instance in self.load(path):
            """ instance
            {
                "conv_id": "hit:3_conv:6",
                "context": "terrified",
                "prompt": "Today,as i was leaving for work in the morning,i had a tire burst in the middle of a busy road. That scared the hell out of me!",
                "dialogue": [
                    ["6", "Today,as i was leaving for work in the morning,i had a tire burst in the middle of a busy road. That scared the hell out of me!"],
                    ["7", "Are you fine now?"],
                    ["6", "Yeah,i'm doing alright now, but with minor injuries."],
                    ...
                ]
            }
            """
            yield [t[-1] for t in instance["dialogue"]]

    def wizard_of_wikipedia(self, path) -> Episode:
        for instance in self.load(path):
            """ instance
            {
                "chosen_topic": "Blue",
                "persona": "my favorite color is blue.",
                "dialog": [
                    ["0_Wizard", "Blue is my favorite primary color."],
                    ["1_Apprentice", "Blue is always nice. I like royal blue."],
                ]
            }
            """
            yield [t[-1] for t in instance["dialog"]]

    def to_parallel(self) -> Generator:
        for dtype, path in self.pathes.items():
            fo_context  = self._dest(path) / self._basename(path, suffix=f"_{self.name}.context")
            fo_response = self._dest(path) / self._basename(path, suffix=f"_{self.name}.response")
            with open(fo_context, "x") as fc, open(fo_response, "x") as fr:
                for episode in self.method(path):
                    for idx in range(1, len(episode)):
                        start = 0 if self.args.use_history else idx-1
                        end = -1 if self.args.use_future else idx+1
                        print(*episode[start:idx], sep=self.sep, file=fc)
                        print(*episode[idx:end], sep=self.sep, file=fr)
                logger.info(f"[write] {fc.name}")
                logger.info(f"[write] {fr.name}")



if __name__ == "__main__":
    """ bash
    python $0 dailydialog
    """
    parser = argparse.ArgumentParser()
    parser = Loader.add_parser(parser)
    args = parser.parse_args()

    loader = Loader(args)
    loader.to_parallel()
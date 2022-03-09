import json
import logging
from os.path import dirname
import sys

sys.path.append(dirname(__file__))
from base import DataFormat

logger = logging.getLogger(__name__)


class DailyDialog(DataFormat):
    def __init__(self, path):
        logger.info("\033[32m" + "DailyDialog" + "\033[0m")
        self.path = path
        self.data = self.load(path)

    def __arxiv__(self):
        return "https://arxiv.org/abs/1710.03957"

    @staticmethod
    def load(path):
        logger.info(f"| LOAD ... {path}")
        dialogues = []
        for instance in open(path):
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
            instance = json.loads(instance.strip())
            dialogue = [t["text"] for t in instance["dialogue"]]
            dialogues.append(dialogue)
        return dialogues
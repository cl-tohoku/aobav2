from os.path import dirname
import sys
sys.path.append(dirname(__file__))

from filters.normalizers import SentenceNormalizer
from filters.parsers import MecabParser
from filters.bert_predictors import NextUtterancePredictor, JsnliPredictor

from dialogs.wiki_template.wiki_template_dialogue import WikipediaTemplateDialogue
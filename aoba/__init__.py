from os.path import dirname
import sys
sys.path.append(dirname(__file__))

from filters.dialog_filter import DialogFilter
from filters.postprocess_scorer import PostprocessScorer
from filters.normalizers import SentenceNormalizer
from filters.parsers import MecabParser
from filters.bert_predictors import NextUtterancePredictor, JsnliPredictor

from giza.evaluate_trans import TransEvaluator
# from giza.sbert import StsEncoder

from dialogs.wiki_template.wiki_template_dialogue import WikipediaTemplateDialogue
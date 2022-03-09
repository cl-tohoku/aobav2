import argparse
import logging
from pprint import pprint
import sys

from elasticsearch import Elasticsearch


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


class MyElasticSearcher(object):
    @classmethod
    def add_parser(cls, parser):
        parser.add_argument_group('Group of ElasticSearch')        
        parser.add_argument("--index_name", required=True, type=str, help="Select index name")
        parser.add_argument("--match_type", choices=["BM25", "EM"], default="BM25", help="Select match type")
        parser.add_argument("--search_target", choices=["title", "text"], default="text", help="Select search target")
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=str, default="9200")
        return parser

    def __init__(self, args):
        self.index_name = args.index_name
        self.ip = f"{args.host}:{args.port}"
        self.match_type = args.match_type
        self.search_target = args.search_target
        self.es = Elasticsearch(self.ip)

    def __del__(self):
        self.es.close()

    def __call__(self, query, top_k=10):
        result = self.es.search(
            index = self.index_name,
            body = self.body(query, self.search_target, self.match_type),
            size = top_k
        )
        outputs = []
        for res in result["hits"]["hits"]:
            outputs.append({
                "title" : res["_source"]["title"], 
                "text" : res["_source"]["text"], 
                "similarity" : res["_score"]
            })
        return outputs

    @staticmethod
    def body(query, search_target, match_type):
        if match_type == "BM25":
            return {
                "query": {
                    "match": {
                        search_target: query
                    }
                }
            }
        else:
            assert search_target == "title",  "EM can be used only for search target \"title\""
            return {
                "query": {
                    "term": {
                        "title.keyword": query
                    }
                }
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser = MyElasticSearcher.add_parser(parser)
    args = parser.parse_args()

    searcher = MyElasticSearcher(args)

    while True:
        try:
            query = str(input('Query > ').strip())
            result = searcher(query)
            print('\033[32m' + f'| Result of {args.match_type}:' + '\033[0m')
            pprint(result)
        except KeyboardInterrupt:
            sys.exit(0)
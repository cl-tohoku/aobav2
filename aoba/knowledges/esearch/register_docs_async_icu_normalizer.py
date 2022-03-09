import argparse
import asyncio
import json
import logging
import sys

from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.helpers import async_streaming_bulk, BulkIndexError


logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_es_index_snippets(index_name, fi_config, delete=False, ip="localhost:9200"):
    index_config = json.load(open(fi_config))
    with Elasticsearch(ip) as es_client:
        if delete and es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
        es_client.indices.create(index=index_name, body=index_config)


async def generate_registration_data(index_name, kb_file):
    for i, line in enumerate(kb_file):
        document = json.loads(line.strip())
        yield {
            "_op_type": "create",
            "_index" : index_name,
            "_source" : document,
            "_id" : i
        }


async def register_document_from_file(index_name, kb_file, ip="localhost:9200"):
    es = AsyncElasticsearch(ip)
    with open(kb_file, mode="r") as fi_kb:
        try:
            async for ok, loop_result in async_streaming_bulk(client=es, actions=generate_registration_data(index_name, fi_kb)):
                action, result = loop_result.popitem()
                logger.info(result)
                if not ok:
                    logger.info(f"failed to {result} document {action}")
        except BulkIndexError as bulk_error:
            logger.warning("\033[31m" + bulk_error.errors + "\033[0m")
    await es.close()


def register(ip, index_name, fi_config, fi_kb, _del=False):
    create_es_index_snippets(index_name, fi_config, delete=_del, ip=ip)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(register_document_from_file(index_name, fi_kb, ip=ip))



if __name__ == "__main__":
    """ bash 
    python $0 <index_name> <knowledge_base_file> \
        [--delete_old_index] \
        [--config_file=index_config.json] \
        [--port=9200] \
        [--host=localhost]

    example:
        - index_name = "wikipedia_kb"
        - knowledge_base_file_path = "./sample_data/mini_sample.json"
    """

    parser = argparse.ArgumentParser(description='ElasticSearch 用に wikipedia のダンプデータをインデクシングするスクリプト')
    parser.add_argument("--index_name", type=str, help="Select index name")
    parser.add_argument("--kb_file", type=str, help="Select path of knowledge base file")
    parser.add_argument("-d", "--delete_old_index", action="store_true")
    parser.add_argument("--config_file", type=str, default="index_config.json", help="The config file for indexing")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="9200")
    args = parser.parse_args()

    ip = f"{args.host}:{args.port}"
    register(ip, args.index_name, args.config_file, args.kb_file, _del=args.delete_old_index)

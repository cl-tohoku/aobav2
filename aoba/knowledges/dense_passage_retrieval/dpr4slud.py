#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import glob
import json
import logging
import os
import pickle
import re
import sys
import time
from tqdm import tqdm
from typing import List, Tuple, Dict, Iterator, Union

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import RepTokenSelector, BiEncoderPassage
from dpr.data.qa_validation import calculate_matches, calculate_chunked_matches
from dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, _select_span_with_token
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)

logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_token_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token)
                        for q in batch_questions
                    ]
                else:
                    batch_token_tensors = [
                        tensorizer.text_to_tensor(" ".join([query_token, q]))
                        for q in batch_questions
                    ]
            else:
                batch_token_tensors = [
                    tensorizer.text_to_tensor(q) for q in batch_questions
                ]

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


def generate_passage_vectors(
    passage_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    passages: List[Union[dict, BiEncoderPassage]],
    bsz: int,
    insert_title: bool = False,
) -> T:
    n = len(passages)
    passage_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_passages = passages[batch_start : batch_start + bsz]
            batch_token_tensors = []
            for ctx in batch_passages:
                if isinstance(ctx, (BiEncoderPassage)):
                    ctx = ctx.__dict__
                batch_token_tensors.append(
                    tensorizer.text_to_tensor(
                        ctx['text'], 
                        title=ctx['title'] if insert_title else None
                    )
                )

            psg_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            psg_seg_batch = torch.zeros_like(psg_ids_batch).cuda()
            psg_attn_mask = tensorizer.get_attn_mask(psg_ids_batch)

            _, out, _ = passage_encoder(psg_ids_batch, psg_seg_batch, psg_attn_mask)

            passage_vectors.extend(out.cpu().split(1, dim=0))

    passage_tensor = torch.cat(passage_vectors, dim=0)
    assert passage_tensor.size(0) == len(passages)
    return passage_tensor


class DenseRetriever(object):
    def __init__(
        self, 
        question_encoder: nn.Module, 
        passage_encoder: nn.Module, 
        batch_size: int, 
        tensorizer: Tensorizer
    ):
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(
        self, questions: List[str], query_token: str = None
    ) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )

    def generate_passage_vectors(
        self, passages: List[Union[dict, BiEncoderPassage]], insert_title: bool = False
    ) -> T:
        bsz = self.batch_size
        self.passage_encoder.eval()
        return generate_passage_vectors(
            self.passage_encoder,
            self.tensorizer,
            passages,
            bsz,
            insert_title=insert_title,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        passage_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, passage_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(
            iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)
        ):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        # self.index = None
        return results


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append(
            {
                "question": q,
                "answers": q_answers,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "title": docs[c][1],
                        "text": docs[c][0],
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(
    vector_files: list, path_id_prefixes: List = None
) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc


def validate_tables(
    passages: Dict[object, TableChunk],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_chunked_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_chunk_hits = match_stats.top_k_chunk_hits
    top_k_table_hits = match_stats.top_k_table_hits

    logger.info("Validation results: top k documents hits %s", top_k_chunk_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_chunk_hits]
    logger.info("Validation results: top k table chunk hits accuracy %s", top_k_hits)

    logger.info("Validation results: top k tables hits %s", top_k_table_hits)
    top_k_table_hits = [v / len(result_ctx_ids) for v in top_k_table_hits]
    logger.info("Validation results: top k tables accuracy %s", top_k_table_hits)

    return match_stats.top_k_chunk_hits



class DenseExtractor():

    def __init__(self, cfg: DictConfig):
        cfg = self.set_config(cfg)
        
        tensorizer, question_encoder, passage_encoder = self.init_model(cfg)
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder
        self.tensorizer = tensorizer

        self.indexer = self.init_indexer(cfg)
        self.retriever = LocalFaissRetriever(
            self.question_encoder, 
            self.passage_encoder,
            cfg.batch_size, 
            self.tensorizer, 
            self.indexer
        )

        self.load_passages(cfg)

    def set_config(self, cfg):
        cfg = setup_cfg_gpu(cfg)
        logger.info("%s", OmegaConf.to_yaml(cfg))
        return cfg

    def init_model(self, cfg):
        saved_state = load_states_from_checkpoint(cfg.model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
        tensorizer, biencoder, _ = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg, inference_only=True
        )
        encoder_path = cfg.encoder_path
        if encoder_path:
            logger.info("Selecting encoder: %s", encoder_path)
            question_encoder = getattr(biencoder, encoder_path)
        else:
            logger.info("Selecting standard question encoder")
            question_encoder = biencoder.question_model
            passage_encoder = biencoder.ctx_model

        question_encoder, _ = setup_for_distributed_mode(
            question_encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
        )
        passage_encoder, _ = setup_for_distributed_mode(
            passage_encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
        )
        question_encoder.eval()
        passage_encoder.eval()

        # load weights from the model file
        question_model_to_load = get_model_obj(question_encoder)
        passage_model_to_load = get_model_obj(passage_encoder)
        logger.info("Loading saved model state ...")
        
        question_encoder_prefix = (encoder_path if encoder_path else "question_model") + "."
        passage_encoder_prefix = (encoder_path if encoder_path else "ctx_model") + "."
        question_prefix_len = len(question_encoder_prefix)
        passage_prefix_len = len(passage_encoder_prefix)

        # logger.info("Encoder state prefix %s", question_encoder_prefix)
        question_encoder_state = {
            key[question_prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(question_encoder_prefix)
        }
        passage_encoder_state = {
            key[passage_prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith(passage_encoder_prefix)
        }
        question_model_to_load.load_state_dict(question_encoder_state, strict=False)
        passage_model_to_load.load_state_dict(passage_encoder_state, strict=False)
        self.vector_size = question_model_to_load.get_out_size()
        logger.info("Question encoder vector_size=%d", self.vector_size)

        return tensorizer, question_encoder, passage_encoder

    def init_indexer(self, cfg):
        indexer = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Index class %s ", type(indexer))
        indexer.init_index(self.vector_size)
        return indexer

    def load_passages(self, cfg):
        id_prefixes = []    # ['wiki:']
        ctx_sources = []
        all_passages = {}
        for ctx_src in cfg.ctx_datatsets:
            """ cfg.ctx_datasets    # conf/ctx_sources/jawiki.yaml
            {
                'dpr_wiki': {
                    '_target': 'dpr.data.retriever_data.CsvCtxSrc',
                    'file': passage_file,
                    'id_prefix': 'wiki:'
                }
            }
            """
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix)
            ctx_sources.append(ctx_src)
            ctx_src.load_data_to(all_passages)
        self.all_passages = all_passages
        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # index all passages
        ctx_files_patterns = cfg.encoded_ctx_files  # [embeddings_file_patterns]
        index_path = cfg.index_path                 # [index_file]
        logger.info("ctx_files_patterns: %s", ctx_files_patterns)
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(id_prefixes)

        input_paths = []        # [embeddings_file]
        path_id_prefixes = []   # ['wiki:']
        for i, pattern in enumerate(ctx_files_patterns):
            pattern_files = glob.glob(pattern)
            pattern_id_prefix = id_prefixes[i]
            input_paths.extend(pattern_files)
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        
        self.index_embeddings_to_faiss(
            index_path=index_path, 
            input_paths=input_paths, 
            path_id_prefixes=path_id_prefixes
        )

    def index_embeddings_to_faiss(self, index_path:str=None, input_paths:List[str]=None, path_id_prefixes:List[str]=None):
        if index_path and self.indexer.index_exists(index_path):
            logger.info("Index path: %s", index_path)
            retriever.index.deserialize(index_path)
        elif input_paths:
            logger.info("Reading all passages data from files: %s", input_paths)
            self.retriever.index_encoded_data(
                input_paths, self.indexer.buffer_size, path_id_prefixes=path_id_prefixes
            )
            if index_path:
                self.retriever.index.serialize(index_path)
                logging.info("Serialize index to: %s", index_path)
        else:
            raise TypeError('index_embeddings_to_faiss() must be any(index_path, input_paths)')

    def encode_query(self, questions:Union[str, List[str]]) -> T: #TensorType['batch', 'dim':768]:
        if isinstance(questions, str):
            questions = [questions]
        questions_tensor = self.retriever.generate_question_vectors(
            questions, query_token=None,
        )
        return questions_tensor
    
    def encode_passage(self, passages) -> T: #TensorType['batch', 'dim':768]:
        if isinstance(passages, (dict, BiEncoderPassage)):
            passages = [passages]
        elif isinstance(passages, str):
            passages = [{'text': passages}]
        passages_tensor = self.retriever.generate_passage_vectors(
            passages, insert_title=False,
        )
        return passages_tensor

    def search_passages(self, question_tensor:T, n_docs:int):
        # question_tensor: TensorType['batch':1, 'dim':768]
        top_ids_and_scores = self.retriever.get_top_docs(question_tensor.numpy(), n_docs)
        return top_ids_and_scores

    def __call__(self, query:str, n_docs:int=100) -> List[dict]:
        question_tensor = self.encode_query(query)
        top_ids_and_scores = self.search_passages(question_tensor, n_docs)
        ids, scores = top_ids_and_scores[0]
        passages = []
        for top_k, (idx, score) in enumerate(zip(ids, scores), start=1):
            passage = self.all_passages[idx]
            passages.append({
                'id': idx,
                'score': score,
                'title': passage.title,
                'text': passage.text,
            })
        return passages
    
    def search_for_file(self, fi_context, fi_response, fo_jsl, batch:int=1024, n_docs:int=8) -> List[dict]:
        buf, tmp = [], []
        with open(fo_jsl, 'w') as fo:
            print(f'| READ ... {fi_response}')
            print(f'| WRITE ... {fo.name}')
            for con, res in tqdm(zip(open(fi_context), open(fi_response))):
                tgt = ''.join(re.sub('\<ST[0-9]+?\>', '', res).strip().split())
                if len(buf) < batch:
                    buf.append(tgt)
                    tmp.append((con, res))
                    continue
                question_tensors = self.encode_query(buf)
                top_ids_and_scores = self.search_passages(question_tensors, n_docs)
                for bix, (ids, scores) in enumerate(top_ids_and_scores):
                    passages = {}
                    passages['context'] = tmp[bix][0]
                    passages['response'] = tmp[bix][1]
                    passages['wiki'] = []
                    for top_k, (idx, score) in enumerate(zip(ids, scores), start=1):
                        passage = self.all_passages[idx]
                        passages['wiki'].append({
                            'score': float(score),
                            'text': passage.text,
                        })
                    fo.write(json.dumps(passages, ensure_ascii=False) + '\n')
                buf = []
            print(f'| WRITE ... {fo.name}')


def main():
    # cfg = OmegaConf.load(open('interact_retriever.yaml'))
    cfg = OmegaConf.load(open('/work02/SLUD2021/models/dpr/interact_retriever.yaml'))
    dense_extractor = DenseExtractor(cfg)
    
    while True:
        try:
            query = str(input('\033[32m' + 'In: Query > ' + '\033[0m'))
            retrieved_passages = dense_extractor(query, n_docs=5)
            for top_k, passage in enumerate(retrieved_passages, start=1):
                print()
                print('\033[34m' + f'Out[{top_k}]: Title > ' + '\033[0m', passage['title'], passage['score'])
                print('\033[34m' + f'Out[{top_k}]: Text  > ' + '\033[0m', passage['text'])

            query_tensor = dense_extractor.encode_query(query)
            passage_tensor = dense_extractor.encode_passage(retrieved_passages[0]['text'])
            passage_tensors = dense_extractor.encode_passage(retrieved_passages)
            import ipdb; ipdb.set_trace()

        except KeyboardInterrupt:
            sys.exit(0)


def extract_related_passages_for_response():
    cfg = OmegaConf.load(open('interact_retriever.yaml'))
    context_file = "/work02/SLUD2021/datasets/dialogue/ja/twitter/pretraining/train_4.5M_w_ne.context"
    response_file = "/work02/SLUD2021/datasets/dialogue/ja/twitter/pretraining/train_4.5M_w_ne.response"
    out_file = response_file.replace('.response', '_response_w_knowledge.json')
    dense_extractor = DenseExtractor(cfg)
    dense_extractor.search_for_file(context_file, response_file, out_file)


if __name__ == "__main__":
    # extract_related_passages_for_response()
    main()

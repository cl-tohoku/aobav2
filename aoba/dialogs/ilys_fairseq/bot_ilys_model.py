#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

from tokenizer import SpmTokenizer



logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")

class IlysFairseqModel:
    def __init__(self, args: Namespace):
        self.args = args
        self.cfg = convert_namespace_to_omegaconf(self.args)
        self._load_args(self.cfg, self.args)
        self.task = tasks.setup_task(self.cfg.task)
        self.bpe = self.task.build_bpe(self.cfg.bpe)
        self.tokenizer = SpmTokenizer(args.spm)
        self.align_dict = utils.load_align_dict(self.cfg.generation.replace_unk)
        self.models = self._load_models(self.cfg, self.args, self.task)
        self.max_positions = self._load_max_positions(self.models, self.task)
        self.generator = self.task.build_generator(self.models, self.cfg.generation)
        self.usecuda = True
    
    @classmethod
    def add_parser(cls, parser):
        parser.add_argument('--spm', type=os.path.abspath, default="/work02/SLUD2021/models/spm/8M_tweets.cr9999.bpe.32000.model", metavar="FP", help="Path to sentencepiece model")
        return parser

    def __call__(self, contexts: List[str], prefix: Union[str, List[str]] = None) -> List[str]:
        tokenized_contexts = [self.tokenizer.encode(utt) for utt in contexts]
        if self.bpe is not None:
            tokenized_contexts = [self.bpe.encode(utt) for utt in contexts]
        context_as_input = '<ST2> ' + ' <s> '.join(tokenized_contexts)
        inputs = [context_as_input]
        logger.info("fairseq context:\n{}".format("\n".join(inputs)))

        # decode
        responses = self.decode(inputs=inputs)
        logger.info("fairseq responses: \n{}".format("\n".join("{}: {}".format(s, text) for s, text in responses)))
        responses = [text for s, text in responses]

        return responses

    def create_batches(self, inputs: List[str]) -> Iterable[Batch]:
        constraints_tensor = None

        tokens = [self.task.source_dictionary.encode_line(line, add_if_not_exist=False).long() for line in inputs]
        lengths = [t.numel() for t in tokens]
        itr = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(
                tokens, lengths, constraints=constraints_tensor
            ),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=self.max_positions,
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
        ).next_epoch_itr(shuffle=False)
        for batch in itr:
            ids = batch["id"]
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            constraints = batch.get("constraints", None)

            yield Batch(
                ids=ids,
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                constraints=constraints,
            )

    def decode(self, inputs: List[str]) -> List[Tuple[float, str]]:
        results = []
        for batch in self.create_batches(inputs):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if self.args.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translate_start_time = time.time()
            translations = self.task.inference_step(
                self.generator, self.models, sample, constraints=constraints
            )
            list_constraints = [[] for _ in range(bsz)]
            if self.cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.task.target_dictionary.pad())
                constraints = list_constraints[i]
                results.append((id, src_tokens_i, hypos))
        responses = []
        # sort output to match input order
        for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            src_str = ''
            if self.task.source_dictionary is not None:
                src_str = self.task.source_dictionary.string(src_tokens, self.cfg.common_eval.post_process)

            # Process top predictions
            for hypo in hypos[: min(len(hypos), self.cfg.generation.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=self.align_dict,
                    tgt_dict=self.task.target_dictionary,
                    remove_bpe=self.cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )
                #print('hopy', hypo_str)
                detok_hypo_str = self.tokenizer.decode(hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                responses.append((float(score), detok_hypo_str))

        return responses

    @staticmethod
    def _load_args(cfg: Namespace, args: Namespace) -> None:

        utils.import_user_module(cfg.common)

        if cfg.interactive.buffer_size < 1:
            cfg.interactive.buffer_size = 1
        if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
            cfg.dataset.batch_size = 1

        assert (
                not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
        ), "--sampling requires --nbest to be equal to --beam"
        assert (
                not cfg.dataset.batch_size
                or cfg.dataset.batch_size <= cfg.interactive.buffer_size
        ), "--batch-size cannot be larger than --buffer-size"

        logger.info(cfg)

        # Fix seed for stochastic decoding
        if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
            np.random.seed(cfg.common.seed)
            utils.set_torch_seed(cfg.common.seed)

        args.use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    @staticmethod
    def _load_models(cfg: Namespace, args: Namespace, task):
        # Load ensemble
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

        for model in models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if args.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)

        return models

    @staticmethod
    def _load_max_positions(models, task):
        max_positions = utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        )
        return max_positions


def send_to_cuda(device_index: int, tensor):
    if device_index is not None:
        return tensor.to(torch.device(device_index))
    else:
        return tensor.cuda()




if __name__ == "__main__":
    parser = options.get_interactive_generation_parser()
    parser = IlysFairseqModel.add_parser(parser)
    args = options.parse_args_and_arch(parser)

    decoder = IlysFairseqModel(args)
    contexts = ["こんにちは"]
    responses = decoder(contexts)

    print(responses)

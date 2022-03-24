#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import json
import math
import os
from argparse import Namespace
from collections import namedtuple
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data import encoders
from fairseq_cli.generate import get_symbols_to_strip_from_output
from logzero import logger

from bot_tokenizer import SpmTokenizer
from pytorch_utils import send_to_cuda

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


class FairSeqModel:
    """
    Usage:
        >> parser = options.get_interactive_generation_parser()
        >> parser.add_argument('--spm', type=os.path.abspath, help="Path to sentencepiece model")
        >> args = options.parse_args_and_arch(parser)

        >> fairseq_model = FairSeqModel(args)
        >> response: str = fairseq_model(['こんにちは'])
    """

    def __init__(self, args: Namespace):
        self.args = args
        self._load_args(self.args)
        self.bpe = encoders.build_bpe(self.args)
        self.task = tasks.setup_task(self.args)
        self.tokenizer = SpmTokenizer(self.args.spm)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)
        self.models = self._load_models(self.args, self.task)
        self.max_positions = self._load_max_positions(self.models, self.task)
        self.generator = self.task.build_generator(self.models, self.args)
        device_index = 0 if self.args.device_fairseq is None else self.args.device_fairseq
        logger.info(f"The fairseq model(s) has been loaded to 'device {device_index}'")

    def __call__(self, contexts: List[str], prefix: Union[str, List[str]] = None) -> List[str]:
        """
        Args:
            contexts: List of utterances
            prefix: The string to prepend the model's input (default=None)
        Return:
            response: Response which is created by the model
        """
        # Create context as input
        tokenized_contexts = [self.tokenizer.encode(utt) for utt in contexts]
        if self.bpe is not None:
            tokenized_contexts = [self.bpe.encode(utt) for utt in contexts]
        context_as_input = ' <s> '.join(tokenized_contexts)

        if prefix is None:
            inputs = [context_as_input]
        elif isinstance(prefix, str):
            inputs = [prefix + context_as_input]
        elif isinstance(prefix, list):
            inputs = [p + context_as_input for p in prefix]
        else:
            raise ValueError(f"The type of 'prefix' is invalid: {type(prefix)}")
        logger.debug("fairseq context:\n{}".format("\n".join(inputs)))

        # decode
        responses = self.decode(inputs=inputs)
        logger.debug("fairseq responses: \n{}".format("\n".join("{}: {}".format(s, text) for s, text in responses)))
        responses = [text for s, text in responses]

        return responses

    def create_batches(self, inputs: List[str]) -> Iterable[Batch]:
        """Create batch as input"""
        tokens = [self.task.source_dictionary.encode_line(line, add_if_not_exist=False).long() for line in inputs]
        lengths = [t.numel() for t in tokens]
        itr = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self.max_positions,
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test
        ).next_epoch_itr(shuffle=False)

        for batch in itr:
            yield Batch(
                ids=batch['id'],
                src_tokens=batch['net_input']['src_tokens'],
                src_lengths=batch['net_input']['src_lengths'],
            )

    def decode(self, inputs: List[str]) -> List[Tuple[float, str]]:
        """
        Args:
            inputs: List of sequences as input to the model
        Returns:
            responses: The list of (score, response) which is created by the model
        """
        results = []
        for batch in self.create_batches(inputs):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.args.use_cuda:
                src_tokens = send_to_cuda(self.args.device_fairseq, src_tokens)
                src_lengths = send_to_cuda(self.args.device_fairseq, src_lengths)
            sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}
            # import ipdb; ipdb.set_trace()
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.task.target_dictionary.pad())
                results.append((id, src_tokens_i, hypos))

        # sort output to match input order
        responses = []
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.task.source_dictionary is not None:
                src_str = self.task.source_dictionary.string(src_tokens, self.args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.align_dict,
                    tgt_dict=self.task.target_dictionary,
                    remove_bpe=self.args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )

                if self.bpe is not None:
                    hypo_str = self.bpe.decode(hypo_str)
                detok_hypo_str = self.tokenizer.decode(hypo_str)

                score = hypo['score'] / math.log(2)  # convert to base 2
                responses.append((float(score), detok_hypo_str))

        return responses

    @staticmethod
    def _load_args(args: Namespace) -> None:
        # Load args from 'args.json_args'
        args_file_path = getattr(args, "json_args", None)
        if args_file_path:
            assert os.path.exists(args_file_path), f"Not found: {args_file_path}"
            loaded_args = json.load(open(args_file_path))
            for k, v in loaded_args.items():
                setattr(args, k.replace("-", "_"), v)
            logger.info(f"Loaded args from '{args_file_path}'")

        # Settings
        utils.import_user_module(args)
        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1
        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'
        logger.info(args)

        # Fix seed for stochastic decoding
        if args.seed is not None and not args.no_seed_provided:
            np.random.seed(args.seed)
            utils.set_torch_seed(args.seed)

        # Fix and add args option
        args.use_cuda = True if torch.cuda.is_available() and not args.cpu else False
        logger.info("CUDA: {}".format(args.use_cuda))

        if "spm" not in args.__dict__ or args.spm is None:
            raise ValueError("Please add option to parser of 'Argparse': --spm [Path to sentencepiece model]")
        elif not os.path.exists(args.spm):
            raise FileNotFoundError("Sentencepiece model is not found: {}".format(args.spm))
        logger.info("SPM: {}".format(args.spm))

    @staticmethod
    def _load_models(args: Namespace, task):
        logger.info(f'loading fairseq model(s) from {args.path}')
        models, _model_args = checkpoint_utils.load_model_ensemble(
            args.path.split(os.pathsep),
            arg_overrides=eval(args.model_overrides),
            task=task,
            suffix=getattr(args, "checkpoint_suffix", ""),
        )

        for model in models:
            model.prepare_for_inference_(args)
            if args.fp16:
                model.half()
            if args.use_cuda:
                send_to_cuda(args.device_fairseq, model)

        return models

    @staticmethod
    def _load_max_positions(models, task):
        max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        )

        return max_positions

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
from pprint import pprint
import os
import sys

from omegaconf import DictConfig, OmegaConf

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler

sys.path.append(os.path.join(os.path.dirname(__file__)))
import fid.slurm
import fid.util
from fid.options import Options
import fid.data
import fid.evaluation
import fid.model

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from knowledges import DenseExtractor


class FidModel(object):
    @classmethod
    def add_parser(cls, parser):
        parser.add_argument('--fid_config', default=os.path.join(os.path.dirname(__file__), 'configs/client_fid.json'))
        return parser
 
    def __init__(self, args):
        args = self.__postinit__(args)
        self.args = args
        self.model = self.init_model(args)
        self.tokenizer = self.init_toker(args)
        self.collator_function = fid.data.Collator(self.tokenizer, args.text_maxlength)

    def __postinit__(self, _args):
        args = copy.deepcopy(_args)
        config_args = json.load(open(args.fid_config))
        override_keys = {arg[len('--'):].split('=')[0] for arg in sys.argv[1:] if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)

        fid.slurm.init_distributed_mode(args)
        fid.slurm.init_signal_handler()

        dir_path = Path(args.checkpoint_dir)/args.name
        directory_exists = dir_path.exists()
        if args.is_distributed:
            torch.distributed.barrier()
        dir_path.mkdir(parents=True, exist_ok=True)
        return args

    def init_model(self, args):
        model_class = fid.model.FiDT5
        model = model_class.from_pretrained(args.model_path)
        model = model.to(args.device)
        model.eval()
        if hasattr(model, "module"):
            model = model.module
        return model

    def init_toker(self, args):
        model_name = '/work02/SLUD2021/models/fid/sonoisa/t5-base-japanese'
        return transformers.T5Tokenizer.from_pretrained(model_name)

    def __doc__(self):
        return 'https://huggingface.co/transformers/model_doc/t5.html#transformers.T5ForConditionalGeneration'
    
    @classmethod
    def convert_retrieved_psgs(cls, query, retrieved_passages):
        return {
            'index': 0,
            'question': 'question: {}'.format(query),
            'target': '',
            'passages': [
                'title: {title} context: {context}'.format(title=psg['title'], context=psg['text']) \
                for psg in retrieved_passages
            ],
            'scores': torch.tensor([float(psg['score']) for psg in retrieved_passages])
        }

    def filter_response(self, response):
        if (response is None) or ('。' not in response):
            return
        output = []
        for sent in response.split('。'):
            if sent not in output:
                output.append(sent)
        return '。'.join(output)

    def detok_response(self, response):
        """ 句点で終わるようにtruncate """
        if '。' not in response:
            return
        ep = response[::-1].index('。')
        return response[::-1][ep:][::-1]

    def __call__(self, input_data):
        args = self.args
        batch = self.collator_function([input_data])
        responses = []
        with torch.no_grad():
            (idx, _, _, context_ids, context_mask) = batch

            for length in [15, 25]:
                outputs = self.model.generate(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    max_length=length,
                    num_beams=5,
                )

                for k, o in enumerate(outputs):
                    response = self.tokenizer.decode(o, skip_special_tokens=True)
                    response = self.detok_response(response)
                    response = self.filter_response(response)
                    if response is not None:
                        responses.append(response)

        return list(set(responses))



def main():
    parser = argparse.ArgumentParser()
    parser = FidModel.add_parser(parser)
    args = parser.parse_args()
    
    retriever_config = os.path.join(os.path.dirname(__file__), "../../knowledges/dense_passage_retrieval/interact_retriever.yml")
    cfg = OmegaConf.load(open(retriever_config))
    dense_extractor = DenseExtractor(cfg)
    decoder = FidModel(args)
    
    while True:
        try:
            query = str(input('\033[32m' + 'In: Query > ' + '\033[0m'))
            retrieved_passages = dense_extractor(query, n_docs=5)
            input_data = FidModel.convert_retrieved_psgs(query, retrieved_passages)
            responses = decoder(input_data)
            print('\033[34m' + f'Out: Response > ' + '\033[0m', responses)

        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()

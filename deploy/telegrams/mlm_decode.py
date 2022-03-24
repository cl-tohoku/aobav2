# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""

import argparse
from dataclasses import dataclass, field
from os import path
from typing import List, Tuple, Optional

import torch
from logzero import logger
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from pytorch_utils import send_to_cuda

def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=path.abspath, required=True, help="Path to model file")

    return parser


@dataclass
class NextUtterancePredictor:
    model_file: str = field(metadata={"help": "Path to model file (pytorch_model.bin)"})
    model_name: str = field(default="cl-tohoku/bert-base-japanese-whole-word-masking", metadata={"help": "model name"})
    no_cuda: bool = field(default=False, metadata={"help": "The model is loaded to cpu"})
    device: int = field(default=None, metadata={"help": "The cuda device index"})
    config: BertConfig = field(init=False, metadata={"help": "Bert model config"})
    tokenizer: BertTokenizer = field(init=False, repr=False, metadata={"help": "Bert model tokenizer"})
    model: BertForSequenceClassification = field(init=False, repr=False, metadata={"help": "Bert classification model"})

    def __post_init__(self):
        logger.info(f"Loading NextUtterancePredictor from {self.model_file}")
        assert path.exists(self.model_file), f"Not found: {self.model_file}"
        self.no_cuda = True if self.no_cuda or not torch.cuda.is_available() else False
        self.config = AutoConfig.from_pretrained(self.model_name, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        self.model.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))
        if not self.no_cuda:
            self.model = send_to_cuda(self.device, self.model)
            device_index = 0 if self.device is None else self.device
            logger.info(f"NextUtterancePredictor has been loaded to 'device {device_index}'")
        else:
            logger.info(f"NextUtterancePredictor has been loaded to cpu")

    def __call__(self, contexts: List[str], candidates: List[str]) -> Optional[List[Tuple[float, str]]]:
        # --- create batch ---
        contexts_embed_ids = [self.tokenizer.cls_token_id]
        for utt in contexts:
            contexts_embed_ids.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utt)))
            contexts_embed_ids.append(self.tokenizer.sep_token_id)
        len_contexts = len(contexts_embed_ids)

        if len_contexts > self.tokenizer.model_max_length:  # over maximum length
            logger.warning(f"Context is over maximum length: {len_contexts} > {self.tokenizer.model_max_length}")
            return None

        valid_candidates = []
        features = []
        for utt in candidates:
            response_embed_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(utt))
            len_response = len(response_embed_ids)

            if len_response + len_contexts > self.tokenizer.model_max_length:  # over maximum length
                logger.warning(f"Over maximum length ({self.tokenizer.model_max_length}): {utt}")
                continue

            feature = {
                "input_ids": contexts_embed_ids + response_embed_ids + [self.tokenizer.sep_token_id],
                "token_type_ids": [0] * len_contexts + [1] * (len_response + 1),
            }
            valid_candidates.append(utt)
            features.append(feature)

        if len(valid_candidates) == 0:
            logger.warning("All candidates are over maximum length")
            return None

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        # --- prediction ---
        if not self.no_cuda:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = send_to_cuda(self.device, v)

        with torch.no_grad():
            output, *_ = self.model(**batch)

        prediction = torch.softmax(output, dim=1).cpu()[:, 1]
        results = [(float(score), response) for score, response in zip(prediction, valid_candidates)]
        results = sorted(results, reverse=True)

        log_message = "reranked responses:\n"
        log_message += "\n".join(f"{score}, {response}" for score, response in results)
        logger.info(log_message)

        return results


def main():
    parser = create_parser()
    args = parser.parse_args()

    predictor = NextUtterancePredictor(args.model)

    while True:
        n_contexts = int(input("> Please enter the number of contexts (e.g. 2): "))
        contexts = []
        for i in range(n_contexts):
            context = input(f"> Please enter the {i + 1}-th context: ")
            contexts.append(context)

        n_candidates = int(input("> Please enter the number of candidates (e.g. 3): "))
        candidates = []
        for i in range(n_candidates):
            candidate = input(f"> Please enter the {i + 1}-th candidate: ")
            candidates.append(candidate)

        results = predictor(contexts, candidates)
        print("\n".join(f"{s}: {candidate}" for s, candidate in results))

        is_continue = input("> Continue ? (yes/no): ")
        if is_continue != "yes":
            break


if __name__ == "__main__":
    main()

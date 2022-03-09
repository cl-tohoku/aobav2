import argparse
import logging
import os
from pprint import pformat
import sys
from typing import List

import numpy as np

import torch

from transformers import T5Tokenizer, AutoModelForCausalLM, AutoConfig
from pytorch_pretrained_bert import GPT2Config

sys.path.append(os.path.dirname(__file__))
from env import SP_TOKENS, END_OF_TURN_TOKEN, END_OF_TEXT_TOKEN
from lsp_model.modeling_gpt2 import GPT2LMHeadModel
from gpt2_training.train_utils import load_model


logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def generate_response(encoded_ids, tokenizer, model, num_beams):
    return model.generate(
            encoded_ids,
            max_length=100,
            min_length=0,
            do_sample=False,
            num_beams=num_beams,
            temperature=1.0,
            top_k=10,
            top_p=0.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=0,
            num_return_sequences=num_beams,
            fp16=True
            )

class DialoGptModel(object):
    @classmethod
    def add_parser(cls, parser):
        parser.add_argument_group("Group of DialoGPT")
        parser.add_argument("--dgpt_model", type=str, default=f"/work02/SLUD2021/models/dialogpt/GP2-pretrain-step-300000.pkl")
        parser.add_argument("--dgpt_config", type=str, default=f"/work02/SLUD2021/models/dialogpt/config.json")
        parser.add_argument("--dgpt_toker", type=str, default=f"/work02/SLUD2021/models/dialogpt/tokenizer")
        parser.add_argument("--dgpt_max_history", type=int, default=1)
        return parser

    def __init__(self, args):
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu"), n_gpu
        self.device = args.device
        args.fp16 = True
        args.n_gpu = 1
        seed = 2021
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.init_tokenizer(args.dgpt_toker)
        self.init_model(args, args.dgpt_model, args.dgpt_config)

        self.eos_id = self.tokenizer.convert_tokens_to_ids(END_OF_TEXT_TOKEN)
        self._eos_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.sep_id = self.tokenizer.convert_tokens_to_ids("<s>")
        self.detok_id = [-1, self.eos_id, self._eos_id, self.sep_id]

        self.length = 200
        self.max_history = args.dgpt_max_history


    def init_tokenizer(self, toker_name_or_path):
        toker = T5Tokenizer.from_pretrained(toker_name_or_path)
        #toker = AutoTokenizer.from_pretrained(toker_name_or_path)
        logger.info(f"| LOAD tokenizer ... {toker_name_or_path}")
        if not os.path.isfile(toker_name_or_path):
            toker.add_special_tokens({"additional_special_tokens": [END_OF_TURN_TOKEN, END_OF_TEXT_TOKEN] + SP_TOKENS})
            toker.do_lower_case = True
        self.tokenizer = toker
        self.vocab_size = len(self.tokenizer.get_vocab())
        logger.info(f"| The size of vocabulary ... {self.vocab_size}")

    def init_model(self, args, model_path, config_file):
        if os.path.isfile(config_file) and os.path.isfile(model_path):
            self.config = GPT2Config.from_json_file(config_file)
            logger.info(f"| LOAD config_file ... {config_file}:")
            logger.info(f"{pformat(self.config.__dict__)}")
            assert self.config.vocab_size == self.vocab_size
        else:
            # logger.warning(f"| Model file is not found: {model_path}")
            raise FileNotFoundError("This script should be used for trained model.")
            _model = AutoModelForCausalLM.from_pretrained(model_path)
            vocab_size = len
            _model.resize_token_embeddings(self.vocab_size)
            _config = _model.config.__dict__
            _config["vocab_size"] = len(self.tokenizer.get_vocab())
            self.config = GPT2Config.from_dict(_config)
        src_model = load_model(GPT2LMHeadModel(self.config), model_path, args, verbose=True)
        tgt_model = AutoModelForCausalLM.from_pretrained(model_path, config=AutoConfig.from_pretrained(config_file))
        src_state = src_model.state_dict()
        tgt_state = tgt_model.state_dict()
        for key, value in src_state.items():
            if key == "lm_head.decoder.weight":
                src_state["lm_head.weight"] = src_state.pop(key)
            elif key not in tgt_model.state_dict():
                print(f"{key} not in AutoModelForCausalLM")
                src_state.pop(key)
        for key, value in tgt_state.items():
            if "attn.masked_bias" in key:
                src_state[key] = tgt_state[key] # "attn.masked_bias が src_model には存在しない"
        tgt_model.load_state_dict(src_state)
        tgt_model.half()
        self.model = tgt_model
        self.model.to(self.device)
        self.model.eval()

    def encode(self, history:List[str]):
        context_tokens = []
        for h in history:
            context_tokens += self.tokenizer.encode(h, add_special_tokens=False) + [self.sep_id]
        context_tokens = context_tokens[:-1] + [self._eos_id, self.eos_id]
        input_ids = torch.tensor(context_tokens, device=self.device, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=self.device)
        return input_ids, position_ids
    
    def __call__(self, history, num_beams=5):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        history = history[-(2*self.max_history)+1:]
        _history = history.copy()
        _history[0] = f"<ST2> {history[0]}"
        input_ids, position_ids = self.encode(_history)
        print(f"Input > {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
        token_type_ids = None
        past = None
        output = input_ids.new_zeros([input_ids.size(0),0])
        prev = input_ids
        try:
            output_ids = generate_response(input_ids, self.tokenizer, self.model, num_beams)
            output_list = [
                self.tokenizer.decode(output_ids[sid], skip_special_tokens=False).strip() \
                for sid in range(len(output_ids))
            ]
            output_list = [
                s.replace("</s>", "").replace(" ", "").split("<|endoftext|>")[-1] \
                for s in output_list
            ]
            return output_list
        except:
            return None


def run():
    parser = argparse.ArgumentParser(description="")
    parser = DialoGptModel.add_parser(parser)
    args = parser.parse_args()

    predictor = DialoGptModel(args)

    history = []
    while True:

        try:
            context = input("\033[34m" + "Input > " + "\033[0m")
            history.append(context)

            response = predictor(history, num_beams=5)
            history.append(response[0])
            print("\033[32m" + f"| Response ... {response}" + "\033[0m")

        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    run()


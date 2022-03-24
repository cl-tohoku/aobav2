#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import copy
import pickle
import random
import socket
from typing import Tuple, List, Union

from logzero import logger

from aoba import (
    SentenceNormalizer,
    WikipediaTemplateDialogue,
)

from postprocess import post_processing, re_ranking_result
from . import (
    UserContexts, SpecialToken, 
    DialogueBotForSocket,
    NextUtterancePredictor,
    PostprocessScorer,
    FairSeqModel
)



class DialogueAgent(object):
    def __init__(self, args: Namespace):
        self.bot_name = "あおば"
        self.args = args
        
        # socket 用
        self.host = args.host_address
        self.client_socks = {}

        # モデルの定義
        # self.fairseq_model: FairSeqModel = FairSeqModel(args)
        self.next_utterance_predictor = NextUtterancePredictor(
            model_file=self.args.mlm_model,
            device=self.args.device_mlm
        )
        self.postprocess_scorer = PostprocessScorer(args)
        self.client_wiki = WikipediaTemplateDialogue(args)
        # self.fairseq_ft_model: FairSeqModel = self.set_other_model(args)

        # DialogueBotForSocket
        self.client_dialogpt = self.set_client("dialogpt")
        self.client_fid = self.set_client("fid")
        self.client_nttcs = self.set_client("nttcs")
        self.client_ilys = self.set_client("ilys")
        logger.info("DialogueModel has been loaded !")

        # normalizer
        self.normalizer = SentenceNormalizer()

    @property
    def _client(self):
        args = self.args
        return {
            "dialogpt": {
                "attr": "client_dialogpt",
                "port": args.dgpt_port,
            },
            "fid": {
                "attr": "client_fid",
                "port": args.fid_port,
            },
            "nttcs": {
                "attr": "client_nttcs",
                "port": args.ilys_port,
            },
            "ilys": {
                "attr": "client_ilys",
                "port": args.ilys_port,
            },
        }

    def set_client(self, client_name):
        def set_server(self, port:int):
            serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            serversock.bind((self.host, port))
            serversock.listen(10)
            return serversock
        logger.info(f"|--> Waiting for a connection from {client_name} model ::: {self.host}")
        server_socket = set_server(self._client[client_name]["port"])
        client_socket, client_address = server_socket.accept()
        logger.info(f"|--> {client_name} was accepted!")
        self.client_sockss[client_name] = client_socket
        return DialogueBotForSocket(client_socket)

    def response_client(self, client_name, user_contexts, start=-1) -> List[str]:
        try:
            model = getattr(self, self._client[client_name]["attr"])
            socket_data = pickle.dumps({'response': user_contexts.contexts[start:], 'prob':0})
            model.send_utter(socket_data)
            received_data = model.receive_utter()
            if received_data is not None:
                received_data = pickle.loads(received_data)
                response = received_data['response']
                if (response is not None) and isinstance(response, list):
                    logger.info(f"{client_name} response: {response}")
                    return response
        except Exception as e:
            logger.error(f"client_{client_name} didn't work. Please make sure the code.")
            logger.error(f"{e.__class__.__name__}: {e}")

    def set_other_model(self, args):
        other_args = copy.deepcopy(args)
        other_args.data = args.other_data
        other_args.json_args = args.other_json_args
        return FairSeqModel(other_args)

    def reload(self) -> None:
        """ Reload fairseq models (fairseq args) """
        # for model in self.fairseq_model.models:
        #     model.cpu()
        #     del model
        self.next_utterance_predictor.model.cpu()
        # self.fairseq_model = FairSeqModel(self.args)
        # self.topic_extractor: TopicExtractor = TopicExtractor(args)
        self.next_utterance_predictor = NextUtterancePredictor(
            model_file=self.args.mlm_model,
            device=self.args.device_mlm
        )
        self.postprocess_scorer = PostprocessScorer(self.args)
        self.wikipedia_template = WikipediaTemplateDialogue(self.args)
        self.client_dialogpt: DialogueBotForSocket = DialogueBotForSocket(self.client_socks['dialogpt'])
        self.client_fid: DialogueBotForSocket = DialogueBotForSocket(self.client_socks['fid'])
        self.client_nttcs: DialogueBotForSocket = DialogueBotForSocket(self.client_socks['nttcs'])
        self.fairseq_ft_model: FairSeqModel = self.set_other_model(self.args)
        logger.info("DialogueModel has been reloaded !")

    @property
    def filter_targets(self):
        return ("♪", "♩", "っ!")

    def filter_fn(self, responses):
        filtered_responses = []
        for response in filter(lambda x: x is not None, responses):
            response = self.normalizer.normalize(response)
            response = self.normalizer.parentheses(response)
            for tgt in self.filter_targets:
                response.replace(tgt, "")
            if response is not None:
                filtered_responses.append(response)
        return filtered_responses

    def first_message(self, user_contexts):
        response = "{user_name}さん、初めまして！{bot_name}と申します。{message}".format(
            user_name = user_contexts.user_name,
            bot_name = self.bot_name,
            message = self.args.start_message
        )
        logger.info("first message (response): {}".format(response))
        # first message is not joined to contexts
        user_contexts.n_turns += 1
        user_contexts.full_contexts.append(response)
        user_contexts.full_users.append("Dialogue Model")

        self.response_client("dialogpt", ["/start"])
        self.response_client("fid", ["/start"])
        self.response_client("nttcs", ["/start"])
        self.response_client("ilys", ["/start"])
        return response, user_contexts

    def __call__(self, utterance: str, user_contexts: UserContexts) -> Tuple[str, UserContexts]:
        """
        Args:
            utterance: The user utterance
            user_contexts: UserContexts
        Return:
            (response, UserContexts)
        """
        logger.info("--------- No. {} --------\n{}".format(user_contexts.n_turns, user_contexts))

        if utterance == "" and user_contexts.n_turns == 0:
            self.first_message(user_contexts)

        logger.info("context: {}".format(utterance))
        user_contexts.add_context(utterance, user_contexts.user_name)  # add utterance to user contexts

        if user_contexts.n_turns == 30:
            response = "ありがとうございました！"
            user_contexts.add_context(response, "Dialogue Model")  # add utterance to user contexts
            return response, user_contexts

        ##########
        # start
        ##########

        responses = []

        # response from client_wiki
        try:
            outputs = self.client_wiki(user_contexts.contexts[-1])
            if outputs is not None:
                _, response = outputs
                logger.info(f"wiki response: {response}")
                responses.append(response)
        except Exception as e:
            logger.error("client_wiki didn't work. Please make sure the code.")
            logger.error(f"{e.__class__.__name__}: {e}")

        # response from clients
        responses.extend(self.response_client("dialogpt", user_contexts, start=-1))
        responses.extend(self.response_client("fid", user_contexts, start=0))
        responses.extend(self.response_client("nttcs", user_contexts, start=-1))
        responses.extend(self.response_client("ilys", user_contexts, start=-1))

        """
        # response from fairseq
        fairseq_responses = self.fairseq_model(user_contexts.contexts)
        responses.extend(fairseq_responses)
        
        # response from change_of_topic
        cot_prefixes = self._create_cot_prefixes(utterance=utterance, user_contexts=user_contexts)
        if cot_prefixes:
            cot_responses = self.fairseq_model(user_contexts.contexts[-1:], cot_prefixes)
            responses.extend(cot_responses)
        """

        # filtering
        responses = self.filter_fn(responses)

        # re-ranking by language model
        re_ranking_results = [
            re_ranking_result(
                full_contexts=user_contexts.full_contexts,
                contexts=user_contexts.contexts,
                response=response,
                next_utterance_score=next_utterance_score
            )
            for next_utterance_score, response in self.next_utterance_predictor(user_contexts.contexts, responses)
        ]

        # post-processing
        response = post_processing(
            re_ranking_results=re_ranking_results,
            postprocess_scorer=self.postprocess_scorer,
            args=self.args,
        )

        user_contexts.add_context(response, "Dialogue Model")  # add utterance to user contexts

        # post-processing 2 (not joined to contexts)
        response = response.replace("あなた", f"{user_contexts.user_name}さん")  # The user's name can affect to context
        if user_contexts.n_turns == 29:
            response += f"って、気づいたらもうこんな時間！{user_contexts.user_name}さんとお話しできて楽しかったです！",
        user_contexts.full_contexts[-1] = response  # update full contexts

        logger.info("response: {}".format(response))
        return response, user_contexts

    def _create_cot_prefixes(self, utterance: str, user_contexts: UserContexts) -> List[str]:
        """
        Create prefix for change of topics.
        This prefix is used for input sequence of fairseq model.
        Args:
            utterance: The user utterance
            user_contexts: UserContexts
        Return:
            prefixes
        """
        cot_prefixes = []

        # Extract topic from sentence
        topic = self.topic_extractor.calc_cos_sim(utterance.rstrip("\n"), self.args.top_n_topic)
        topic_word_frequency = self.topic_extractor.word2cnt[topic]
        logger.info(f"Topic: {topic} ({topic_word_frequency}), <- {utterance}")

        # Add topic to list
        if self.args.word_freq_min_threshold <= topic_word_frequency <= self.args.word_freq_max_threshold:
            topic_tokenized = self.fairseq_model.tokenizer.encode(topic)
            user_contexts.topics.append((topic_word_frequency, topic_tokenized))

        # Decide whether to change the topic
        if (
                len(user_contexts.topics) != 0
                and user_contexts.n_turns >= self.args.min_cot_contexts
                and random.random() <= self.args.cot_prob
        ):
            for selected_word_freq, selected_topic in user_contexts.topics:
                cot_prefix = f"{selected_topic} {SpecialToken.COT.value} "
                cot_prefixes.append(cot_prefix)

        return cot_prefixes

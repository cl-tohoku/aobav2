#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
from argparse import Namespace
from typing import Tuple, List

from logzero import logger

from .. import UserContexts

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
END = '\033[0m'

typeUtterance = str
typeResponse = str


class BaseDialogueAgent(object):
    def __init__(self):
        self.bot_name = "あおば"
        self.user_name = ""
        self.model_name = "Dialogue Model"
        self.end_turn = 30

    def set_user(self, user_contexts: UserContexts):
        """ UserContexts からユーザ情報を登録 """
        self.user_name = user_contexts.user_name

    @property
    def first_response(self) -> typeResponse:
        return "{user_name}さん、初めまして！{bot_name}と申します。{message}".format(
            user_name=self.user_name,
            bot_name=self.bot_name,
            message="今日は何のお話をしましょう？"
        )

    @property
    def before_last_response(self) -> typeResponse:
        return "って、気づいたらもうこんな時間！{user_name}さんとお話しできて楽しかったです！".format(
            user_name=self.user_name
        )

    @property
    def last_response(self) -> typeResponse:
        return "ありがとうございました！"

    @abstractmethod
    def get_response_candidates(self, user_contexts: UserContexts) -> List[typeResponse]:
        """ 応答候補 List[typeResponse] を取得 """
        utterance = user_contexts.full_contexts[-1]
        return [utterance]

    @abstractmethod
    def select_response(self, response_candidates:List[typeResponse]) -> typeResponse:
        """ 応答候補 List[typeResponse] から最終出力を一つ選択  """
        return response_candidates[0]

    def __call__(self, utterance: typeUtterance, user_contexts: UserContexts) -> Tuple[typeResponse, UserContexts]:
        """
        Args:
            - utterance (typeUtterance): ユーザからの発話
            - user_contexts (UserContexts): 対話履歴を含む UserContext

        Return:
            - typeResponse: ユーザ発話に対するモデルからの応答
            - UserContexts: 更新された UserContext
        """
        self.set_user(user_contexts)    # TODO: socket で注意
        logger.info(f"--------- No. {user_contexts.n_turns} --------")
        # logger.info(f"* bot_name:  {user_contexts.bot_name}")
        # logger.info(f"* user_name: {user_contexts.user_name}")
        for idx, (user, context) in enumerate(zip(user_contexts.full_users, user_contexts.full_contexts)):
            logger.info(f"No.{idx} [{user}] {context}")

        # 対話開始時の処理
        if utterance == "" and user_contexts.n_turns == 0:
            response = self.first_response
            logger.info("first message (response): {}".format(response))
            user_contexts.n_turns += 1
            user_contexts.full_contexts.append(response)
            user_contexts.full_users.append(self.model_name)
            return response, user_contexts

        # UserContext を更新: ユーザ発話を追加
        user_contexts.add_context(utterance, user_contexts.user_name)
        logger.info(f"No.{len(user_contexts)-1} [{self.user_name}] {utterance}")    # 0 start

        ##################################################
        # モデルからの応答を取得
        ##################################################

        if user_contexts.n_turns < self.end_turn:
            # 応答候補の取得
            response_candidates: List[typeResponse] = self.get_response_candidates(user_contexts)

            # 応答候補から一つを選択
            response: typeResponse = self.select_response(response_candidates)

            # 対話終了一つ前
            if user_contexts.n_turns == self.end_turn-1:
                response += self.before_last_response

            # UserContext を更新: モデル応答を追加
            user_contexts.add_context(response, self.model_name)
            logger.info(
                YELLOW 
                + f"No.{len(user_contexts)-1} [{self.model_name}] {utterance}    #selected response"
                + END
            )

            return response, user_contexts

        elif user_contexts.n_turns == self.end_turn:
            logger.info("response: {}".format(self.last_response))
            user_contexts.add_context(self.last_response, self.model_name)
            return response, user_contexts

        else:
            raise NotImplementedError("The termination processing was not executed successfully.")

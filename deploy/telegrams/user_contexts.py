#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from typing import List, Tuple



@dataclasses.dataclass
class DialogueInstance:
    turn: int
    user: str
    utterance: str
    def __str__(self):
        return f"No.{self.turn} [{self.user}] {self.utterance}"


class UserContexts:
    def __init__(self, bot_name:str, user_id:int, first_name:str, last_name:str, max_len_contexts:int=0):
        self.bot_name = bot_name
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.user_name = first_name + " " + last_name
        self.max_len_contexts: int = max_len_contexts
        assert type(self.max_len_contexts) == int and self.max_len_contexts >= 0
        self.initialize()

    def __repr__(self) -> str:
        print(self.__dict__)
        return "\n".join("{}: {}".format(k, v) for k, v in self.__dict__.items())

    def __len__(self) -> int:
        return len(self.full_contexts[-self.max_len_contexts:])

    def __getitem__(self, idx:int) -> DialogueInstance:
        return self.full_contexts[idx]

    def add_context(self, utterance:str, user:str) -> None:
        # self.contexts.append(utterance)
        # self.users.append(user)
        # self.contexts = self.contexts[-self.max_len_contexts:]
        # self.users = self.users[-self.max_len_contexts:]
        self.n_turns += 1
        self.full_contexts.append(
            DialogueInstance(
                turn=self.n_turns,
                user=user,
                utterance=utterance,
            )
        )

    def initialize(self) -> None:
        #self.contexts: List[str] = []
        #self.users: List[str] = []
        self.full_contexts: List[DialogueInstance] = []
        self.full_users: List[str] = []
        self.topics: List[Tuple[int, str]] = []
        self.n_turns: int = 0

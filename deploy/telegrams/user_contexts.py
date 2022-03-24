from typing import List, Tuple


class UserContexts:
    def __init__(self, bot_name: str, user_id: int, first_name: str, last_name: str, max_len_contexts: int = 0):
        self.bot_name: str = bot_name
        self.user_id: int = user_id
        self.first_name: str = first_name
        self.last_name: str = last_name
        self.max_len_contexts: int = max_len_contexts
        assert type(self.max_len_contexts) == int and self.max_len_contexts >= 0

        self.user_name: str = "{} {}".format(self.first_name, self.last_name)
        self.contexts: List[str] = []
        self.users: List[str] = []
        self.full_contexts: List[str] = []
        self.full_users: List[str] = []
        self.topics: List[Tuple[int, str]] = []
        self.n_turns: int = 0

    def __repr__(self) -> str:
        return "\n".join("{}: {}".format(k, v) for k, v in self.__dict__.items())

    def __len__(self) -> int:
        return len(self.contexts)

    def add_context(self, utterance: str, user: str) -> None:
        self.contexts.append(utterance)
        self.users.append(user)
        self.contexts = self.contexts[-self.max_len_contexts:]
        self.users = self.users[-self.max_len_contexts:]
        self.n_turns += 1
        assert len(self.contexts) == len(self.users)

        self.full_contexts.append(utterance)
        self.full_users.append(user)

    def reset(self) -> None:
        self.contexts = []
        self.users = []
        self.full_contexts = []
        self.full_users = []
        self.topics = []
        self.n_turns = 0
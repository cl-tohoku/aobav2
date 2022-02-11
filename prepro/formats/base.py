import abc
from typing import List


class DataFormat(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, path):
        Dialogue = List[str]
        self.data: List[Dialogue] = [[]]
    
    def __len__(self, ):
        return len(self.data)

    def __getitem__(self, idx:int):
        return self.data[idx]

    @abc.abstractmethod
    def load(self, fi_data:str):
        raise NotImplementedError()
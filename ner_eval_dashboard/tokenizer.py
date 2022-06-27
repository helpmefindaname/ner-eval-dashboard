import abc

from ner_eval_dashboard.datamodels import PreTokenizedText, Text


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, test: Text) -> PreTokenizedText:
        raise NotImplementedError()

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()

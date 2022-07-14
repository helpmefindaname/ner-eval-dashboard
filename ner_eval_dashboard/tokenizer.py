import abc

from ner_eval_dashboard.datamodels import PreTokenizedText, Text
from ner_eval_dashboard.utils import RegisterMixin, setup_register


@setup_register
class Tokenizer(abc.ABC, RegisterMixin):
    @abc.abstractmethod
    def tokenize(self, test: Text) -> PreTokenizedText:
        raise NotImplementedError()

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()

import abc
from typing import List, Sequence

from ner_eval_dashboard.datamodels import LabeledText, PreTokenizedText, Text, Token
from ner_eval_dashboard.utils import RegisterMixin, setup_register


@setup_register
class Tokenizer(abc.ABC, RegisterMixin):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def tokenize(self, text: Text) -> PreTokenizedText:
        raise NotImplementedError()

    def hash(self) -> str:
        return self.__class__.__name__

    def tokenize_labeled_seq(self, texts: Sequence[LabeledText]) -> List[PreTokenizedText]:
        return [self.tokenize(text) for text in texts]


@Tokenizer.register("SPACE")
class SpaceTokenizer(Tokenizer):
    def tokenize(self, text: Text) -> PreTokenizedText:
        text_tokens = text.text.split(" ")
        start = 0
        tokens = []
        for text_token in text_tokens:
            if text_token == "":
                continue
            start += text.text[start:].index(text_token)
            end = start + len(text_token)
            tokens.append(Token(start=start, end=end, text=text_token))

            start = end

        return PreTokenizedText(dataset_type=text.dataset_type, dataset_text_id=text.dataset_text_id, tokens=tokens)

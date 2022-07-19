from pathlib import Path
from typing import List, Optional, Set

from flair.data import Sentence, Span
from flair.data import Token as FlairToken
from flair.models import SequenceTagger

from ner_eval_dashboard.datamodels import Label, LabeledTokenizedText, PreTokenizedText
from ner_eval_dashboard.predictor import Predictor


@Predictor.register("FLAIR")
class FlairPredictor(Predictor):
    def __init__(self, name_or_path: str):
        super().__init__()
        self.tagger: SequenceTagger = SequenceTagger.load(name_or_path)
        possible_path = Path(name_or_path)
        if possible_path.exists():
            if possible_path.stem in ["pytorch_model", "final-model", "best-model"]:
                self._name = possible_path.parent.stem
            else:
                self._name = possible_path.stem
        else:
            self._name = name_or_path
        self._label_names: Optional[List[str]] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def label_names(self) -> List[str]:
        if self._label_names is not None:
            return self._label_names

        labels: Set[str] = set()

        for label in self.tagger.label_dictionary.get_items():
            if label[:2] in ["B-", "I-", "E-", "S-"]:
                labels.add(label[2:])

        self._label_names = sorted(labels)

        return self._label_names

    def predict(self, data: List[PreTokenizedText]) -> List[LabeledTokenizedText]:
        sentences = list(map(self._tokenized_text_to_sentence, data))
        self.tagger.predict(sentences)

        labels = map(self._sentence_to_labels, sentences)
        return self._combine_with_labels(data, labels)

    def _sentence_to_labels(self, sentence: Sentence) -> List[Label]:
        return [
            Label(
                entity_type=span.tag,
                start=span.start_position,
                end=span.end_position,
                text=self._get_span_text(span),
            )
            for span in sentence.get_spans(self.tagger.label_type)
        ]

    @staticmethod
    def _get_span_text(span: Span) -> str:
        text = ""
        for token in span.tokens:
            text += token.text
            text += " " * token.whitespace_after
        return text.strip()

    @staticmethod
    def _tokenized_text_to_sentence(text: PreTokenizedText) -> Sentence:
        sentence = Sentence(text=[])
        previous_token: Optional[FlairToken] = None
        for token in text.tokens:
            flair_token = FlairToken(token.text, start_position=token.start)
            sentence.add_token(flair_token)
            if previous_token is not None:
                previous_token.whitespace_after = flair_token.start_pos - previous_token.end_pos
            previous_token = flair_token

        return sentence

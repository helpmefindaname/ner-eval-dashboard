import warnings
from pathlib import Path
from typing import List, Optional, Set

from flair.data import Sentence, Span
from flair.data import Token as FlairToken
from flair.models import SequenceTagger

from ner_eval_dashboard.datamodels import (
    Label,
    LabeledTokenizedText,
    PreTokenizedText,
    ScoredLabel,
    ScoredTokenizedText,
)
from ner_eval_dashboard.predictor import Predictor
from ner_eval_dashboard.predictor_mixins import (
    DropoutPredictorMixin,
    ScoredTokenPredictorMixin,
)


@Predictor.register("FLAIR")
class FlairPredictor(ScoredTokenPredictorMixin, DropoutPredictorMixin, Predictor):
    def __init__(self, name_or_path: str):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
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

    @property
    def tag_label_names(self) -> List[str]:
        return sorted(self.tagger.label_dictionary.get_items())

    def predict(self, data: List[PreTokenizedText]) -> List[LabeledTokenizedText]:
        sentences = list(map(self._tokenized_text_to_sentence, data))
        self.tagger.eval()
        self.tagger.predict(sentences, verbose=True)

        labels = map(self._sentence_to_labels, sentences)
        return self._combine_with_labels(data, labels)

    def predict_with_dropout(self, data: List[PreTokenizedText]) -> List[LabeledTokenizedText]:
        sentences = list(map(self._tokenized_text_to_sentence, data))
        self.tagger.train()
        self.tagger.predict(sentences, verbose=True)

        labels = map(self._sentence_to_labels, sentences)
        return self._combine_with_labels(data, labels)

    def predict_token_scores(self, data: List[PreTokenizedText]) -> List[ScoredTokenizedText]:
        sentences = list(map(self._tokenized_text_to_sentence, data))
        self.tagger.eval()
        self.tagger.predict(sentences, return_probabilities_for_all_classes=True, verbose=True)

        scores = map(self._sentence_to_scores, sentences)
        return self._combine_with_scores(data, scores)

    def _sentence_to_labels(self, sentence: Sentence) -> List[Label]:
        return [
            Label.construct(
                entity_type=span.tag,
                start=span.start_position,
                end=span.end_position,
                text=self._get_span_text(span),
            )
            for span in sentence.get_spans(self.tagger.label_type)
        ]

    def _sentence_to_scores(self, sentence: Sentence) -> List[List[ScoredLabel]]:
        return [
            [
                ScoredLabel.construct(
                    tag=label.value,
                    score=label.score,
                )
                for label in token.get_tags_proba_dist(self.tagger.label_type)
            ]
            for token in sentence.tokens
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
        previous_token: Optional[FlairToken] = None
        tokens: List[FlairToken] = []
        for token in text.tokens:
            flair_token = FlairToken(token.text, start_position=token.start)
            if previous_token is not None:
                previous_token.whitespace_after = flair_token.start_position - previous_token.end_position
            previous_token = flair_token
            tokens.append(flair_token)
        sentence = Sentence(text=tokens)

        return sentence

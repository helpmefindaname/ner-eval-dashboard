import abc
from typing import TYPE_CHECKING, Iterable, List, Type

from ner_eval_dashboard.component import (
    ExplainaboardComponent,
    F1MetricComponent,
    TestExamplesComponent,
    TrainingExamplesComponent,
    UnlabeledPredictionExamplesComponent,
    ValidationExamplesComponent,
)
from ner_eval_dashboard.datamodels import (
    Label,
    LabeledTokenizedText,
    PreTokenizedText,
    ScoredLabel,
    ScoredToken,
    ScoredTokenizedText,
)
from ner_eval_dashboard.utils import RegisterMixin, setup_register
from ner_eval_dashboard.utils.hash import json_hash

if TYPE_CHECKING:
    from ner_eval_dashboard.component import Component


@setup_register
class Predictor(abc.ABC, RegisterMixin):
    def __init__(self) -> None:
        self._components: List[Type["Component"]] = []
        self.add_component(F1MetricComponent)
        self.add_component(TrainingExamplesComponent)
        self.add_component(ValidationExamplesComponent)
        self.add_component(TestExamplesComponent)
        self.add_component(UnlabeledPredictionExamplesComponent)
        self.add_component(ExplainaboardComponent)

    def add_component(self, component: Type["Component"]) -> None:
        self.components.append(component)

    @property
    def components(self) -> List[Type["Component"]]:
        return self._components

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def label_names(self) -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, data: List[PreTokenizedText]) -> List[LabeledTokenizedText]:
        raise NotImplementedError()

    def hash(self) -> str:
        return json_hash([self.name, self.label_names])

    @staticmethod
    def _combine_with_labels(
        data: Iterable[PreTokenizedText], labels: Iterable[List[Label]]
    ) -> List[LabeledTokenizedText]:
        return [
            LabeledTokenizedText.model_construct(
                tokens=text.tokens,
                dataset_type=text.dataset_type,
                dataset_text_id=text.dataset_text_id,
                labels=label,
            )
            for text, label in zip(data, labels)
        ]

    @staticmethod
    def _combine_with_scores(
        data: Iterable[PreTokenizedText], data_scores: Iterable[List[List[ScoredLabel]]]
    ) -> List[ScoredTokenizedText]:
        return [
            ScoredTokenizedText.model_construct(
                tokens=[
                    ScoredToken.model_construct(
                        start=token.start,
                        end=token.end,
                        text=token.text,
                        scored_labels=token_scores,
                    )
                    for token, token_scores in zip(text.tokens, scores)
                ],
                dataset_type=text.dataset_type,
                dataset_text_id=text.dataset_text_id,
            )
            for text, scores in zip(data, data_scores)
        ]

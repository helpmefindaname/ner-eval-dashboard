import abc
from typing import TYPE_CHECKING, Iterable, List, Type

from ner_eval_dashboard.component import F1MetricComponent
from ner_eval_dashboard.datamodels import Label, LabeledTokenizedText, PreTokenizedText
from ner_eval_dashboard.utils import RegisterMixin, setup_register
from ner_eval_dashboard.utils.hash import json_hash

if TYPE_CHECKING:
    from ner_eval_dashboard.component import Component


@setup_register
class Predictor(abc.ABC, RegisterMixin):
    def __init__(self) -> None:
        self._components: List[Type["Component"]] = []
        self.add_component(F1MetricComponent)

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
            LabeledTokenizedText(
                tokens=text.tokens,
                dataset_type=text.dataset_type,
                dataset_text_id=text.dataset_text_id,
                labels=label,
            )
            for text, label in zip(data, labels)
        ]


class PredictorMixin:
    def add_component(self, component: Type["Component"]) -> None:
        super_obj = super(self)
        add_fn = getattr(super_obj, "add_component", None)
        if add_fn is not None:
            add_fn(component)

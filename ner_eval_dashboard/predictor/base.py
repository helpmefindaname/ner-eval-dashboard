import abc
from typing import TYPE_CHECKING, List, Type

from ner_eval_dashboard.datamodels import LabeledTokenizedText, PreTokenizedText

if TYPE_CHECKING:
    from ner_eval_dashboard.component import Component


class Predictor(abc.ABC):
    def __init__(self) -> None:
        self._components: List[Type["Component"]] = []

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

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError()


class PredictorMixin:
    def add_component(self, component: Type["Component"]) -> None:
        super_obj = super(self)
        add_fn = getattr(super_obj, "add_component", None)
        if add_fn is not None:
            add_fn(component)

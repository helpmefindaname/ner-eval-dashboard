import abc
import typing
from typing import List, Type

from ner_eval_dashboard.datamodels import LabeledTokenizedText, PreTokenizedText

if typing.TYPE_CHECKING:
    from ner_eval_dashboard.component import Component


class PredictorMixin:
    def add_component(self, component: Type["Component"]) -> None:
        super_obj = super(PredictorMixin, self)
        add_fn = getattr(super_obj, "add_component", None)
        if add_fn is not None:
            add_fn(component)


class DropoutPredictorMixin(PredictorMixin):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        from ner_eval_dashboard.component import DropoutRobustnessComponent

        super(DropoutPredictorMixin, self).__init__(*args, **kwargs)
        self.add_component(DropoutRobustnessComponent)

    @abc.abstractmethod
    def predict_with_dropout(self, data: List[PreTokenizedText]) -> List[LabeledTokenizedText]:
        raise NotImplementedError()

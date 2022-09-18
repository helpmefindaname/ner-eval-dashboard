import abc
import typing
from typing import List, Type

from ner_eval_dashboard.datamodels import (
    LabeledTokenizedText,
    PreTokenizedText,
    ScoredTokenizedText,
)

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


class ScoredTokenPredictorMixin(PredictorMixin):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super(ScoredTokenPredictorMixin, self).__init__(*args, **kwargs)

        from ner_eval_dashboard.component import PerTokenStats
        self.add_component(PerTokenStats)

    @property
    @abc.abstractmethod
    def tag_label_names(self) -> List[str]:
        raise NotImplementedError()

    @property
    def token_format(self) -> typing.Literal["BIO", "BILOU", "BIOES"]:
        tag_types = set(t[0] for t in self.tag_label_names if len(t) > 2 and t[1] == "-")
        if "S" in tag_types or "E" in tag_types:
            return "BIOES"
        if "L" in tag_types or "U" in tag_types:
            return "BILOU"
        return "BIO"

    @abc.abstractmethod
    def predict_token_scores(self, data: List[PreTokenizedText]) -> List[ScoredTokenizedText]:
        raise NotImplementedError()

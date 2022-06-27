import abc
from typing import TYPE_CHECKING, Any, Dict

from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.cache import delete_cache, has_cache, load_cache
from ner_eval_dashboard.datamodels import SectionType
from ner_eval_dashboard.dataset import Dataset

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


class Component(abc.ABC):
    component_name: str

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        pass

    @classmethod
    def create(cls, predictor: "Predictor", dataset: Dataset) -> "Component":
        key = cls.hash_key(predictor, dataset)
        if has_cache(key):
            data = load_cache(key)
            try:
                return cls(**data)
            except Exception:
                delete_cache(key)
        data = cls.precompute(predictor, dataset)
        return cls(**data)

    @classmethod
    def hash_key(cls, predictor: "Predictor", dataset: Dataset) -> str:
        return bin(hash((predictor, dataset, cls.component_name)))

    @classmethod
    @abc.abstractmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dash_component(self) -> DashComponent:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def section_type(self) -> SectionType:
        raise NotImplementedError()

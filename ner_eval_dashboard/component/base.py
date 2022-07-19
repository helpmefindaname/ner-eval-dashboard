import abc
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from dash.development.base_component import Component as DashComponent
from loguru import logger

from ner_eval_dashboard.cache import delete_cache, has_cache, load_cache, save_cache
from ner_eval_dashboard.datamodels import DatasetType, SectionType
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.utils.hash import json_hash

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


class Component(abc.ABC):
    component_name: str
    dataset_requirements: Tuple[DatasetType]

    def __init__(self, **kwargs: dict) -> None:
        pass

    @classmethod
    def create(cls, predictor: "Predictor", dataset: Dataset) -> "Component":
        key = cls.hash_key(predictor, dataset)
        if has_cache(key):
            logger.info(f"Try loading cache for {cls.component_name}")
            data = load_cache(key)
            try:
                return cls(**data)
            except Exception:
                logger.exception(f"Error loading cache for {cls.component_name}: Deleting invalid cache")
                delete_cache(key)
        else:
            logger.debug(f"No cache for `{key}` found.")
        logger.info(f"Computing parameters for {cls.component_name}")
        data = cls.precompute(predictor, dataset)
        save_cache(key, data)
        return cls(**data)

    @classmethod
    def hash_key(cls, predictor: "Predictor", dataset: Dataset) -> str:
        return json_hash([dataset.hash(cls.dataset_requirements), predictor.hash(), cls.component_name])

    @classmethod
    def can_apply(cls, dataset: Dataset) -> bool:
        if DatasetType.TRAIN in cls.dataset_requirements and not dataset.has_train:
            return False
        if DatasetType.VALIDATION in cls.dataset_requirements and not dataset.has_val:
            return False
        if DatasetType.TEST in cls.dataset_requirements and not dataset.has_test:
            return False
        if DatasetType.UNLABELED in cls.dataset_requirements and not dataset.has_unlabeled:
            return False

        return True

    @classmethod
    @abc.abstractmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dash_components(self) -> List[DashComponent]:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def section_type(self) -> SectionType:
        raise NotImplementedError()

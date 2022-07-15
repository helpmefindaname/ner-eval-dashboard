import pytest

from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.tokenizer import SpaceTokenizer

from .test_utils import assert_dataset_standards


@pytest.mark.parametrize("dataset_name", ["WNUT17", "WEIBO", "MOVIE_COMPLEX"])
def test_flair_dataset(dataset_name: str) -> None:
    dataset_cls = Dataset.load(dataset_name)

    dataset: Dataset = dataset_cls(SpaceTokenizer())
    assert dataset.name == dataset_name
    assert dataset.has_train
    assert dataset.has_val
    assert dataset.has_test
    assert not dataset.has_unlabeled

    assert_dataset_standards(dataset)

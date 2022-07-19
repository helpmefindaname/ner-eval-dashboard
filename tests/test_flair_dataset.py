from pathlib import Path
from typing import Callable

import flair
import pytest

from ner_eval_dashboard.dataset import Dataset, FlairColumnDataset, FlairJsonlDataset
from ner_eval_dashboard.tokenizer import SpaceTokenizer
from tests.test_utils import assert_dataset_standards


@pytest.mark.parametrize("dataset_name", ["WNUT17", "WEIBO", "MOVIE_COMPLEX"])
def test_flair_dataset(dataset_name: str) -> None:
    dataset_cls = Dataset.load(dataset_name)

    dataset: Dataset = dataset_cls(SpaceTokenizer())  # typing: ignore
    assert dataset.name == dataset_name
    assert dataset.has_train
    assert dataset.has_val
    assert dataset.has_test
    assert not dataset.has_unlabeled

    assert_dataset_standards(dataset)


@pytest.mark.skipif(flair.__version__ <= "0.11.3", reason="Flair JsonL does not support span labels yet.")
def test_jsonl_dataset(testdata: Callable[[str], Path]) -> None:

    dataset = FlairJsonlDataset(SpaceTokenizer(), str(testdata("jsonl")))
    assert dataset.name == "jsonl"
    assert dataset.label_names == ["LOC", "MISC", "ORG", "PER"]
    assert dataset.has_train
    assert dataset.has_val
    assert dataset.has_test
    assert not dataset.has_unlabeled

    assert_dataset_standards(dataset)


def test_column_corpus_dataset(testdata: Callable[[str], Path]) -> None:

    dataset = FlairColumnDataset(SpaceTokenizer(), str(testdata("column")))
    assert dataset.name == "column"
    assert dataset.label_names == ["LOC"]
    assert dataset.has_train
    assert dataset.has_val
    assert dataset.has_test
    assert not dataset.has_unlabeled

    assert_dataset_standards(dataset)

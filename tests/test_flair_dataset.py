import pytest

from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.tokenizer import SpaceTokenizer


@pytest.mark.parametrize("dataset_name", ["WNUT17", "WEIBO", "MOVIE_COMPLEX"])
def test_flair_dataset(dataset_name: str) -> None:
    dataset_cls = Dataset.load(dataset_name)

    dataset: Dataset = dataset_cls(SpaceTokenizer())
    assert dataset.name == dataset_name
    assert dataset.has_train
    assert dataset.has_val
    assert dataset.has_test
    assert not dataset.has_unlabeled

    assert len(dataset.label_names) > 0
    label_name_set = set(dataset.label_names)
    assert len(label_name_set) == len(dataset.label_names)
    for ds in [dataset._train, dataset._val, dataset._test]:
        for ex in ds:
            for label in ex.labels:
                assert label.entity_type in label_name_set
                assert ex.text[label.start : label.end] == label.text

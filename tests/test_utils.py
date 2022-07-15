from ner_eval_dashboard.dataset import Dataset


def assert_dataset_standards(dataset: Dataset) -> None:
    __tracebackhide__ = True

    assert len(dataset.label_names) > 0
    label_name_set = set(dataset.label_names)
    assert len(label_name_set) == len(dataset.label_names)
    for ds in [dataset._train, dataset._val, dataset._test]:
        for ex in ds:
            for label in ex.labels:
                assert label.entity_type in label_name_set
                assert ex.text[label.start : label.end] == label.text

from typing import Literal

import pytest

from ner_eval_dashboard.dataset.flair import FlairConll03
from ner_eval_dashboard.tokenizer import SpaceTokenizer


@pytest.mark.debug
@pytest.mark.parametrize("tag_format", ["BIO", "BIOES", "BILOU"])
def test_conll_conversion(tag_format: Literal["BIO", "BIOES", "BILOU"]) -> None:
    dataset = FlairConll03(SpaceTokenizer())
    valid_entity_types = ["O"] + [f"{tag}-{label}" for tag in tag_format for label in dataset.label_names]

    for token_labeled, example in zip(dataset.get_test_token_labeled(tag_format), dataset.test_tokenized):
        assert len(token_labeled.tokens) == len(example.tokens)
        last = "O"
        for labeled_token, token in zip(token_labeled.tokens, example.tokens):
            assert labeled_token.text == token.text
            assert labeled_token.start == token.start
            assert labeled_token.end == token.end
            assert labeled_token.entity_type in valid_entity_types

            if labeled_token.entity_type != "O":
                if labeled_token.entity_type[0] in "BUS":
                    if tag_format == "BIOES":
                        assert last[0] in "OES"
                    if tag_format == "BILOU":
                        assert last[0] in "OUL"
                else:
                    assert last[2:] == labeled_token.entity_type[2:]
                    assert last[0] in "BI"
            elif last != "O" and tag_format != "BIO":
                assert last[0] in "ESUL"

            last = labeled_token.entity_type


@pytest.mark.debug
@pytest.mark.parametrize("tag_format", ["BIO", "BIOES", "BILOU"])
def test_conll_conversion_back(tag_format: Literal["BIO", "BIOES", "BILOU"]) -> None:
    dataset = FlairConll03(SpaceTokenizer())
    tagged_examples = dataset.get_test_token_labeled(tag_format)

    if tag_format == "BIO":
        actual_examples = dataset.from_bio(tagged_examples)
    elif tag_format == "BIOES":
        actual_examples = dataset.from_bioes(tagged_examples)
    else:
        actual_examples = dataset.from_bilou(tagged_examples)

    for actual, expected in zip(actual_examples, dataset.test):
        assert actual.labels == expected.labels

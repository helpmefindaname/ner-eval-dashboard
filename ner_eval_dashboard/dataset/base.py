from typing import Callable, Iterator, List, Sequence, Set

from ner_eval_dashboard.datamodels import (
    DatasetType,
    Label,
    LabeledText,
    LabeledTokenizedText,
    Text,
    TokenLabeledText,
)
from ner_eval_dashboard.tokenizer import Tokenizer


def combine_span_tags(tags: List[Label]) -> Label:
    start = tags[0].start
    end = tags[-1].end
    label = tags[0].entity_type[2:]
    last = start
    text = ""
    for tag in tags:
        text += " " * (tag.start - last)
        text += tag.text
        last = tag.end
    return Label.construct(start=start, end=end, entity_type=label, text=text)


def convert_tags_to_labeled_text(
    extractor_fn: Callable[[List[Label]], Iterator[Label]], examples: Sequence[TokenLabeledText]
) -> List[LabeledTokenizedText]:
    results: List[LabeledTokenizedText] = []
    for example in examples:
        tokens = [token.to_token() for token in example.tokens]
        labels = list(extractor_fn(example.tokens))

        results.append(
            LabeledTokenizedText.construct(
                dataset_type=example.dataset_type,
                dataset_text_id=example.dataset_text_id,
                tokens=tokens,
                labels=labels,
            )
        )
    return results


class Dataset:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        train: List[LabeledText],
        val: List[LabeledText],
        test: List[LabeledText],
        unlabeled: List[Text] = None,
    ):
        self.tokenizer = tokenizer
        self._train = train
        self._val = val
        self._test = test
        if unlabeled is not None:
            self._unlabeled = unlabeled
        else:
            self._unlabeled = []
        label_names: Set[str] = set()
        for ds in [train, val, test]:
            for ex in ds:
                for label in ex.labels:
                    label_names.update(label.entity_type)
        self._label_names = sorted(label_names)

    @property
    def label_names(self):
        return self._label_names

    @property
    def has_unlabeled(self):
        return bool(self._unlabeled)

    @property
    def has_train(self):
        return bool(self._train)

    @property
    def has_val(self):
        return bool(self._val)

    @property
    def has_test(self):
        return bool(self._test)

    def add_unlabeled(self, texts: Sequence[str]):
        start_id = len(self._unlabeled)
        self._unlabeled.extend(
            [
                Text.construct(text=text, dataset_text_id=i, dataset_type=DatasetType.UNLABELED)
                for i, text in enumerate(texts, start=start_id)
            ]
        )

    @classmethod
    def from_bio(cls, examples: Sequence[TokenLabeledText]) -> List[LabeledTokenizedText]:
        def convert(token_labels: List[Label]) -> Iterator[Label]:
            is_span = False
            current_tokens: List[Label] = []
            for token_label in token_labels:
                if is_span and token_label.entity_type == f"I-{current_tokens[0].entity_type[2:]}":
                    current_tokens.append(token_label)
                    continue
                assert token_label.entity_type[0] != "I"
                if is_span:
                    yield combine_span_tags(current_tokens)
                    current_tokens = []
                    is_span = False
                if token_label.entity_type[0] == "B":
                    is_span = True
                    current_tokens.append(token_label)

            if is_span:
                yield combine_span_tags(current_tokens)

        return convert_tags_to_labeled_text(convert, examples)

    @classmethod
    def from_bioes(cls, examples: Sequence[TokenLabeledText], s_tag="S", e_tag="E") -> List[LabeledTokenizedText]:
        def convert(token_labels: List[Label]) -> Iterator[Label]:
            is_span = False
            current_tokens: List[Label] = []
            for token_label in token_labels:
                tag = token_label.entity_type[0]
                label = token_label.entity_type[2:]
                if is_span and label == current_tokens[0].entity_type[2:] and tag in ["I", e_tag]:
                    current_tokens.append(token_label)
                    if tag == e_tag:
                        yield combine_span_tags(current_tokens)
                        current_tokens = []
                        is_span = False
                    continue
                assert tag != "I"
                assert not is_span
                if tag == s_tag:
                    yield combine_span_tags([token_label])
                if tag == "B":
                    is_span = True
                    current_tokens.append(token_label)

        return convert_tags_to_labeled_text(convert, examples)

    @classmethod
    def from_bilou(cls, examples: Sequence[TokenLabeledText]) -> List[LabeledTokenizedText]:
        return cls.from_bioes(examples, s_tag="U", e_tag="L")
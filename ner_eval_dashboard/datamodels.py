from collections import defaultdict
from enum import Enum
from typing import DefaultDict, Dict, List, Literal, Optional

import pydantic
from pydantic import BaseModel
from pydantic_core.core_schema import FieldValidationInfo


def default_dict_add(a: Dict[str, int], b: Dict[str, int]) -> DefaultDict[str, int]:
    r: DefaultDict[str, int] = defaultdict(int)
    for k, v in a.items():
        r[k] += v
    for k, v in b.items():
        r[k] += v
    return r


class DatasetType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNLABELED = "unlabeled"


class SectionType(str, Enum):
    BASIC_METRICS = "basic_metrics"
    EXAMPLES = "examples"
    DETAILED_METRICS = "detailed_metrics"
    ROBUSTNESS = "robustness"


class BaseElement(BaseModel):
    dataset_type: DatasetType
    dataset_text_id: int

    model_config = {"frozen": True}

    @property
    def id(self) -> str:
        return f"{self.dataset_type.value.lower()}-{self.dataset_text_id}"


class Token(BaseModel):
    start: int
    end: int
    text: str

    def to_token(self) -> "Token":
        return Token.construct(text=self.text, start=self.start, end=self.end)

    @pydantic.field_validator("text")
    def validate_text_length(cls, field_value: str, info: FieldValidationInfo) -> str:
        start = info.data["start"]
        end = info.data["end"]
        length = len(field_value)

        if length != end - start:
            raise ValueError(f"the length of `text` must be `end` - `start` but is {length} instead of {end-start}")

        return field_value

    model_config = {"frozen": True}


class ScoredLabel(BaseModel):
    tag: str
    score: float

    model_config = {"frozen": True}


class PreTokenizedText(BaseElement):
    tokens: List[Token]

    model_config = {"frozen": True}

    @classmethod
    def from_tokens(cls, words: List[str], text_id: int) -> "PreTokenizedText":
        start = 0
        tokens: List[Token] = []
        for word in words:
            end = start + len(word)
            tokens.append(
                Token.construct(
                    start=start,
                    end=end,
                    text=word,
                )
            )
            start = end + 1
        return PreTokenizedText(tokens=tokens, dataset_type=DatasetType.UNLABELED, dataset_text_id=text_id)


class Text(BaseElement):
    text: str

    model_config = {"frozen": True}


class Label(Token):
    entity_type: str

    model_config = {"frozen": True}


class LabeledTokenizedText(PreTokenizedText):
    labels: List[Label]

    model_config = {"frozen": True}


class TokenLabeledText(BaseElement):
    tokens: List[Label]

    model_config = {"frozen": True}

    @classmethod
    def from_labeled_tokenized_text(
        cls,
        labeled_tokenized_text: LabeledTokenizedText,
        tag_format: Literal["BIO", "BIOES", "BILOU"] = "BIO",
    ) -> "TokenLabeledText":
        current_label: Optional[Label] = None
        labels = iter(labeled_tokenized_text.labels)
        next_label = next(labels, None)
        labeled_tokens = []
        for token in labeled_tokenized_text.tokens:
            if current_label is not None and token.end <= current_label.end:
                middle_or_end_tag = "I"
                if token.end == current_label.end:
                    if tag_format == "BIOES":
                        middle_or_end_tag = "E"
                    if tag_format == "BILOU":
                        middle_or_end_tag = "L"

                labeled_tokens.append(
                    Label.construct(
                        text=token.text,
                        entity_type=f"{middle_or_end_tag}-{current_label.entity_type}",
                        start=token.start,
                        end=token.end,
                    )
                )
                continue
            current_label = None
            if next_label is None or next_label.start > token.end:
                labeled_tokens.append(
                    Label.construct(
                        text=token.text,
                        entity_type="O",
                        start=token.start,
                        end=token.end,
                    )
                )
                continue
            current_label = next_label
            start_tag = "B"
            if current_label.end <= token.end:
                if tag_format == "BIOES":
                    start_tag = "S"
                if tag_format == "BILOU":
                    start_tag = "U"
            labeled_tokens.append(
                Label.construct(
                    text=token.text,
                    entity_type=f"{start_tag}-{current_label.entity_type}",
                    start=token.start,
                    end=token.end,
                )
            )
            next_label = next(labels, None)

        return cls.construct(
            tokens=labeled_tokens,
            dataset_type=labeled_tokenized_text.dataset_type,
            dataset_text_id=labeled_tokenized_text.dataset_text_id,
        )


class LabeledText(Text):
    labels: List[Label]

    model_config = {"frozen": True}

    @classmethod
    def from_labeled_tokenized_text(cls, labeled_tokenized_text: LabeledTokenizedText) -> "LabeledText":
        text = ""
        last = 0
        for token in labeled_tokenized_text.tokens:
            text += " " * (token.start - last)
            text += token.text
            last = token.end
        return cls.construct(
            text=text,
            labels=labeled_tokenized_text.labels,
            dataset_type=labeled_tokenized_text.dataset_type,
            dataset_text_id=labeled_tokenized_text.dataset_text_id,
        )


class ScoredToken(Token):
    scored_labels: List[ScoredLabel]

    model_config = {"frozen": True}

    def get_score_by_tag(self, tag: str) -> float:
        for label in self.scored_labels:
            if label.tag == tag:
                return label.score
        return 0

    def best(self) -> ScoredLabel:
        return max(self.scored_labels, key=lambda s: s.score)


class ScoredTokenizedText(BaseElement):
    tokens: List[ScoredToken]

    model_config = {"frozen": True}


class LabelPredictionText(BaseElement):
    text: str
    predictions: List[Label]
    labels: List[Label]

    model_config = {"frozen": True}

    @classmethod
    def from_prediction_label_pair(
        cls, predictions: LabeledTokenizedText, labels: LabeledText
    ) -> "LabelPredictionText":
        return LabelPredictionText.construct(
            text=labels.text,
            labels=labels.labels,
            predictions=predictions.labels,
            dataset_text_id=labels.dataset_text_id,
            dataset_type=labels.dataset_type,
        )


class ErrorType(str, Enum):
    MATCH = "MATCH"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    PARTIAL_MATCH = "PARTIAL_MATCH"
    PARTIAL_TYPE_MISMATCH = "PARTIAL_TYPE_MISMATCH"
    PARTIAL_FALSE_POSITIVE = "PARTIAL_FALSE_POSITIVE"
    PARTIAL_FALSE_NEGATIVE = "PARTIAL_FALSE_NEGATIVE"
    NONE = "NONE"


class ErrorSpan(BaseModel):
    text: str
    error: ErrorType
    expected: Optional[str]
    predicted: Optional[str]


class PredictionErrorSpans(BaseElement):
    spans: List[ErrorSpan]


class ConfusionScores(BaseModel):
    micro_tp: int = 0
    micro_fp: int = 0
    micro_fn: int = 0
    cls_tps: Dict[str, int] = defaultdict(int)
    cls_fps: Dict[str, int] = defaultdict(int)
    cls_fns: Dict[str, int] = defaultdict(int)
    overlap_tp: int = 0
    overlap_fp: int = 0
    overlap_fn: int = 0

    def __add__(self, other: "ConfusionScores") -> "ConfusionScores":
        self.cls_tps = default_dict_add(self.cls_tps, other.cls_tps)
        self.cls_fps = default_dict_add(self.cls_fps, other.cls_fps)
        self.cls_fns = default_dict_add(self.cls_fns, other.cls_fns)

        self.micro_tp += other.micro_tp
        self.micro_fp += other.micro_fp
        self.micro_fn += other.micro_fn

        self.overlap_tp += other.overlap_tp
        self.overlap_fp += other.overlap_fp
        self.overlap_fn += other.overlap_fn

        return self


class TypeConfusionScores(BaseModel):
    type_tps: Dict[str, int] = defaultdict(int)
    type_fps: Dict[str, int] = defaultdict(int)
    type_fns: Dict[str, int] = defaultdict(int)

    def __add__(self, other: "TypeConfusionScores") -> "TypeConfusionScores":
        self.type_tps = default_dict_add(self.type_tps, other.type_tps)
        self.type_fps = default_dict_add(self.type_fps, other.type_fps)
        self.type_fns = default_dict_add(self.type_fns, other.type_fns)

        return self

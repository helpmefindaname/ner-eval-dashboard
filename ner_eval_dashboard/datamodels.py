from collections import namedtuple
from enum import Enum
from typing import Any, Dict, List, Optional

import pydantic
from pydantic import BaseModel


class DatasetType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNLABELED = "unlabeled"


class SectionType(str, Enum):
    BASIC_METRICS = "basic_metrics"
    EXAMPLES = "examples"


class BaseElement(BaseModel):
    dataset_type: DatasetType
    dataset_text_id: int

    class Config:
        frozen = True

    @property
    def id(self) -> str:
        return f"{self.dataset_type.value.lower()}-{self.dataset_text_id}"


class Token(BaseModel):
    start: int
    end: int
    text: str

    def to_token(self) -> "Token":
        return Token.construct(text=self.text, start=self.start, end=self.end)

    @pydantic.validator("text")
    def validate_text_length(cls, field_value: str, values: Dict[str, Any], field: Any, config: Any) -> str:
        start = values["start"]
        end = values["end"]
        length = len(field_value)

        if length != end - start:
            raise ValueError(f"the length of `text` must be `end` - `start` but is {length} instead of {end-start}")

        return field_value

    class Config:
        frozen = True


class ScoredLabel(BaseModel):
    tag: str
    score: float

    class Config:
        frozen = True


class PreTokenizedText(BaseElement):
    tokens: List[Token]

    class Config:
        frozen = True


class Text(BaseElement):
    text: str

    class Config:
        frozen = True


class Label(Token):
    entity_type: str

    class Config:
        frozen = True


class TokenLabeledText(BaseElement):
    tokens: List[Label]

    class Config:
        frozen = True


class LabeledTokenizedText(PreTokenizedText):
    labels: List[Label]

    class Config:
        frozen = True


class LabeledText(Text):
    labels: List[Label]

    class Config:
        frozen = True

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

    class Config:
        frozen = True


class ScoredTokenizedText(BaseElement):
    tokens: List[ScoredToken]

    class Config:
        frozen = True


class LabelPredictionText(BaseElement):
    text: str
    predictions: List[Label]
    labels: List[Label]

    class Config:
        frozen = True

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


Callback = namedtuple("Callback", ["inputs", "output", "function"])

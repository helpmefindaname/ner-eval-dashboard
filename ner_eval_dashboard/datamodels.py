from enum import Enum
from typing import Any, Dict, List

import pydantic
from pydantic import BaseModel


class DatasetType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNLABELED = "unlabeled"


class SectionType(str, Enum):
    BASIC_METRICS = "basic_metrics"


class BaseElement(BaseModel):
    dataset_type: DatasetType
    dataset_text_id: int

    class Config:
        frozen = True


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


class LabeledText(Text):
    labels: List[Label]

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


class ScoredToken(Token):
    scored_labels: List[ScoredLabel]

    class Config:
        frozen = True


class ScoredTokenizedText(BaseElement):
    tokens: List[ScoredToken]

    class Config:
        frozen = True

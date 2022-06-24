from enum import Enum
from typing import List

import pydantic
from pydantic import BaseModel


class DatasetType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNLABELED = "unlabeled"


class BaseElement(BaseModel):
    dataset_type: DatasetType
    dataset_text_id: int


class Token(BaseModel):
    start: int
    end: int
    text: str

    def to_token(self) -> "Token":
        return Token.construct(text=self.text, start=self.start, end=self.end)

    @pydantic.validator("text")
    def validate_text_length(cls, field_value, values, field, config):
        start = values["start"]
        end = values["end"]
        length = len(field_value)

        if length != end - start:
            raise ValueError(f"the length of `text` must be `end` - `start` but is {length} instead of {end-start}")

        return field_value


class ScoredLabel(BaseModel):
    tag: str
    score: float


class PreTokenizedText(BaseElement):
    tokens: List[Token]


class Text(BaseElement):
    text: str


class Label(Token):
    entity_type: str


class LabeledText(Text):
    labels: List[Label]


class TokenLabeledText(BaseElement):
    tokens: List[Label]


class LabeledTokenizedText(PreTokenizedText):
    labels: List[Label]


class ScoredToken(Token):
    scored_labels: List[ScoredLabel]


class ScoredTokenizedText(BaseElement):
    tokens: List[ScoredToken]

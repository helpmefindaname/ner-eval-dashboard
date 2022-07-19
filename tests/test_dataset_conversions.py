from ner_eval_dashboard.datamodels import (
    DatasetType,
    Label,
    LabeledTokenizedText,
    Token,
    TokenLabeledText,
)
from ner_eval_dashboard.dataset.base import Dataset


def test_bio_to_labels() -> None:
    # I like music.
    # I live in Austria, but I am not Sigmund    Freud and also not Wolfgang Amadeus Mozart
    examples = [
        TokenLabeledText(
            dataset_text_id=0,
            dataset_type=DatasetType.TEST,
            tokens=[
                Label(text="I", start=0, end=1, entity_type="O"),
                Label(text="like", start=2, end=6, entity_type="O"),
                Label(text="musik", start=7, end=12, entity_type="O"),
                Label(text=".", start=12, end=13, entity_type="O"),
            ],
        ),
        TokenLabeledText(
            dataset_text_id=1,
            dataset_type=DatasetType.TEST,
            tokens=[
                Label(text="I", start=0, end=1, entity_type="O"),
                Label(text="live", start=2, end=6, entity_type="O"),
                Label(text="in", start=7, end=9, entity_type="O"),
                Label(text="Austria", start=10, end=17, entity_type="B-LOC"),
                Label(text=",", start=17, end=18, entity_type="O"),
                Label(text="but", start=19, end=22, entity_type="O"),
                Label(text="I", start=23, end=24, entity_type="O"),
                Label(text="am", start=25, end=27, entity_type="O"),
                Label(text="not", start=28, end=31, entity_type="O"),
                Label(text="Sigmund", start=32, end=39, entity_type="B-PER"),
                Label(text="Freud", start=43, end=48, entity_type="I-PER"),
                Label(text="and", start=49, end=52, entity_type="O"),
                Label(text="also", start=53, end=57, entity_type="O"),
                Label(text="not", start=58, end=61, entity_type="O"),
                Label(text="Wolfgang", start=62, end=70, entity_type="B-PER"),
                Label(text="Amadeus", start=71, end=78, entity_type="I-PER"),
                Label(text="Mozart", start=79, end=85, entity_type="I-PER"),
            ],
        ),
    ]

    expected = [
        LabeledTokenizedText(
            dataset_text_id=0,
            dataset_type=DatasetType.TEST,
            tokens=[
                Token(text="I", start=0, end=1),
                Token(text="like", start=2, end=6),
                Token(text="musik", start=7, end=12),
                Token(text=".", start=12, end=13),
            ],
            labels=[],
        ),
        LabeledTokenizedText(
            dataset_text_id=1,
            dataset_type=DatasetType.TEST,
            tokens=[
                Token(text="I", start=0, end=1),
                Token(text="live", start=2, end=6),
                Token(text="in", start=7, end=9),
                Token(text="Austria", start=10, end=17),
                Token(text=",", start=17, end=18),
                Token(text="but", start=19, end=22),
                Token(text="I", start=23, end=24),
                Token(text="am", start=25, end=27),
                Token(text="not", start=28, end=31),
                Token(text="Sigmund", start=32, end=39),
                Token(text="Freud", start=43, end=48),
                Token(text="and", start=49, end=52),
                Token(text="also", start=53, end=57),
                Token(text="not", start=58, end=61),
                Token(text="Wolfgang", start=62, end=70),
                Token(text="Amadeus", start=71, end=78),
                Token(text="Mozart", start=79, end=85),
            ],
            labels=[
                Label(text="Austria", start=10, end=17, entity_type="LOC"),
                Label(text="Sigmund    Freud", start=32, end=48, entity_type="PER"),
                Label(text="Wolfgang Amadeus Mozart", start=62, end=85, entity_type="PER"),
            ],
        ),
    ]

    result = Dataset.from_bio(examples)

    assert result == expected


def test_bioes_to_labels() -> None:
    # I like music.
    # I live in Austria, but I am not Sigmund    Freud and also not Wolfgang Amadeus Mozart
    examples = [
        TokenLabeledText(
            dataset_text_id=0,
            dataset_type=DatasetType.TEST,
            tokens=[
                Label(text="I", start=0, end=1, entity_type="O"),
                Label(text="like", start=2, end=6, entity_type="O"),
                Label(text="musik", start=7, end=12, entity_type="O"),
                Label(text=".", start=12, end=13, entity_type="O"),
            ],
        ),
        TokenLabeledText(
            dataset_text_id=1,
            dataset_type=DatasetType.TEST,
            tokens=[
                Label(text="I", start=0, end=1, entity_type="O"),
                Label(text="live", start=2, end=6, entity_type="O"),
                Label(text="in", start=7, end=9, entity_type="O"),
                Label(text="Austria", start=10, end=17, entity_type="S-LOC"),
                Label(text=",", start=17, end=18, entity_type="O"),
                Label(text="but", start=19, end=22, entity_type="O"),
                Label(text="I", start=23, end=24, entity_type="O"),
                Label(text="am", start=25, end=27, entity_type="O"),
                Label(text="not", start=28, end=31, entity_type="O"),
                Label(text="Sigmund", start=32, end=39, entity_type="B-PER"),
                Label(text="Freud", start=43, end=48, entity_type="E-PER"),
                Label(text="and", start=49, end=52, entity_type="O"),
                Label(text="also", start=53, end=57, entity_type="O"),
                Label(text="not", start=58, end=61, entity_type="O"),
                Label(text="Wolfgang", start=62, end=70, entity_type="B-PER"),
                Label(text="Amadeus", start=71, end=78, entity_type="I-PER"),
                Label(text="Mozart", start=79, end=85, entity_type="E-PER"),
            ],
        ),
    ]

    expected = [
        LabeledTokenizedText(
            dataset_text_id=0,
            dataset_type=DatasetType.TEST,
            tokens=[
                Token(text="I", start=0, end=1),
                Token(text="like", start=2, end=6),
                Token(text="musik", start=7, end=12),
                Token(text=".", start=12, end=13),
            ],
            labels=[],
        ),
        LabeledTokenizedText(
            dataset_text_id=1,
            dataset_type=DatasetType.TEST,
            tokens=[
                Token(text="I", start=0, end=1),
                Token(text="live", start=2, end=6),
                Token(text="in", start=7, end=9),
                Token(text="Austria", start=10, end=17),
                Token(text=",", start=17, end=18),
                Token(text="but", start=19, end=22),
                Token(text="I", start=23, end=24),
                Token(text="am", start=25, end=27),
                Token(text="not", start=28, end=31),
                Token(text="Sigmund", start=32, end=39),
                Token(text="Freud", start=43, end=48),
                Token(text="and", start=49, end=52),
                Token(text="also", start=53, end=57),
                Token(text="not", start=58, end=61),
                Token(text="Wolfgang", start=62, end=70),
                Token(text="Amadeus", start=71, end=78),
                Token(text="Mozart", start=79, end=85),
            ],
            labels=[
                Label(text="Austria", start=10, end=17, entity_type="LOC"),
                Label(text="Sigmund    Freud", start=32, end=48, entity_type="PER"),
                Label(text="Wolfgang Amadeus Mozart", start=62, end=85, entity_type="PER"),
            ],
        ),
    ]

    result = Dataset.from_bioes(examples)

    assert result == expected


def test_bilou_to_labels() -> None:
    # I like music.
    # I live in Austria, but I am not Sigmund    Freud and also not Wolfgang Amadeus Mozart
    examples = [
        TokenLabeledText(
            dataset_text_id=0,
            dataset_type=DatasetType.TEST,
            tokens=[
                Label(text="I", start=0, end=1, entity_type="O"),
                Label(text="like", start=2, end=6, entity_type="O"),
                Label(text="musik", start=7, end=12, entity_type="O"),
                Label(text=".", start=12, end=13, entity_type="O"),
            ],
        ),
        TokenLabeledText(
            dataset_text_id=1,
            dataset_type=DatasetType.TEST,
            tokens=[
                Label(text="I", start=0, end=1, entity_type="O"),
                Label(text="live", start=2, end=6, entity_type="O"),
                Label(text="in", start=7, end=9, entity_type="O"),
                Label(text="Austria", start=10, end=17, entity_type="U-LOC"),
                Label(text=",", start=17, end=18, entity_type="O"),
                Label(text="but", start=19, end=22, entity_type="O"),
                Label(text="I", start=23, end=24, entity_type="O"),
                Label(text="am", start=25, end=27, entity_type="O"),
                Label(text="not", start=28, end=31, entity_type="O"),
                Label(text="Sigmund", start=32, end=39, entity_type="B-PER"),
                Label(text="Freud", start=43, end=48, entity_type="L-PER"),
                Label(text="and", start=49, end=52, entity_type="O"),
                Label(text="also", start=53, end=57, entity_type="O"),
                Label(text="not", start=58, end=61, entity_type="O"),
                Label(text="Wolfgang", start=62, end=70, entity_type="B-PER"),
                Label(text="Amadeus", start=71, end=78, entity_type="I-PER"),
                Label(text="Mozart", start=79, end=85, entity_type="L-PER"),
            ],
        ),
    ]

    expected = [
        LabeledTokenizedText(
            dataset_text_id=0,
            dataset_type=DatasetType.TEST,
            tokens=[
                Token(text="I", start=0, end=1),
                Token(text="like", start=2, end=6),
                Token(text="musik", start=7, end=12),
                Token(text=".", start=12, end=13),
            ],
            labels=[],
        ),
        LabeledTokenizedText(
            dataset_text_id=1,
            dataset_type=DatasetType.TEST,
            tokens=[
                Token(text="I", start=0, end=1),
                Token(text="live", start=2, end=6),
                Token(text="in", start=7, end=9),
                Token(text="Austria", start=10, end=17),
                Token(text=",", start=17, end=18),
                Token(text="but", start=19, end=22),
                Token(text="I", start=23, end=24),
                Token(text="am", start=25, end=27),
                Token(text="not", start=28, end=31),
                Token(text="Sigmund", start=32, end=39),
                Token(text="Freud", start=43, end=48),
                Token(text="and", start=49, end=52),
                Token(text="also", start=53, end=57),
                Token(text="not", start=58, end=61),
                Token(text="Wolfgang", start=62, end=70),
                Token(text="Amadeus", start=71, end=78),
                Token(text="Mozart", start=79, end=85),
            ],
            labels=[
                Label(text="Austria", start=10, end=17, entity_type="LOC"),
                Label(text="Sigmund    Freud", start=32, end=48, entity_type="PER"),
                Label(text="Wolfgang Amadeus Mozart", start=62, end=85, entity_type="PER"),
            ],
        ),
    ]

    result = Dataset.from_bilou(examples)

    assert result == expected

from ner_eval_dashboard.datamodels import (
    DatasetType,
    LabeledTokenizedText,
    PreTokenizedText,
    Token,
)
from ner_eval_dashboard.predictor import FlairPredictor, Predictor


def test_flair_predictor_predict() -> None:
    predictor = FlairPredictor("helpmefindaname/mini-sequence-tagger-conll03")

    assert predictor.name == "helpmefindaname/mini-sequence-tagger-conll03"
    assert predictor.label_names == ["LOC", "MISC", "ORG", "PER"]

    examples = [
        PreTokenizedText(
            dataset_text_id=0,
            dataset_type=DatasetType.TEST,
            tokens=[
                Token(text="I", start=0, end=1),
                Token(text="like", start=2, end=6),
                Token(text="musik", start=7, end=12),
                Token(text=".", start=12, end=13),
            ],
        ),
        PreTokenizedText(
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
            labels=[],
        ),
    ]

    assert predictor.predict(examples) == expected


def test_flair_predictor_registered() -> None:
    assert Predictor.load("FLAIR") == FlairPredictor

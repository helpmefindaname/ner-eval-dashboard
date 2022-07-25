from ner_eval_dashboard.component.examples import create_prediction_error_span
from ner_eval_dashboard.datamodels import (
    DatasetType,
    ErrorSpan,
    ErrorType,
    Label,
    LabelPredictionText,
    PredictionErrorSpans,
)


def test_match() -> None:

    parameter = LabelPredictionText(
        text="I like Vienna.",
        labels=[Label(text="Vienna", start=7, end=13, entity_type="LOCATION")],
        predictions=[Label(text="Vienna", start=7, end=13, entity_type="LOCATION")],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    expected = PredictionErrorSpans(
        spans=[
            ErrorSpan(text="I like ", error=ErrorType.NONE, expected=None, predicted=None),
            ErrorSpan(
                text="Vienna",
                error=ErrorType.MATCH,
                expected="LOCATION",
                predicted="LOCATION",
            ),
            ErrorSpan(
                text=".",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
        ],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    assert create_prediction_error_span(parameter) == expected


def test_mismatch() -> None:

    parameter = LabelPredictionText(
        text="I like Vienna.",
        labels=[Label(text="Vienna", start=7, end=13, entity_type="LOCATION")],
        predictions=[Label(text="Vienna", start=7, end=13, entity_type="PERSON")],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    expected = PredictionErrorSpans(
        spans=[
            ErrorSpan(text="I like ", error=ErrorType.NONE, expected=None, predicted=None),
            ErrorSpan(
                text="Vienna",
                error=ErrorType.TYPE_MISMATCH,
                expected="LOCATION",
                predicted="PERSON",
            ),
            ErrorSpan(
                text=".",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
        ],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    assert create_prediction_error_span(parameter) == expected


def test_false_negative() -> None:

    parameter = LabelPredictionText(
        text="I like Vienna.",
        labels=[Label(text="Vienna", start=7, end=13, entity_type="LOCATION")],
        predictions=[],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    expected = PredictionErrorSpans(
        spans=[
            ErrorSpan(text="I like ", error=ErrorType.NONE, expected=None, predicted=None),
            ErrorSpan(
                text="Vienna",
                error=ErrorType.FALSE_NEGATIVE,
                expected="LOCATION",
                predicted=None,
            ),
            ErrorSpan(
                text=".",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
        ],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    assert create_prediction_error_span(parameter) == expected


def test_false_positive() -> None:

    parameter = LabelPredictionText(
        text="I like Music.",
        labels=[],
        predictions=[Label(text="Music", start=7, end=12, entity_type="LOCATION")],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    expected = PredictionErrorSpans(
        spans=[
            ErrorSpan(text="I like ", error=ErrorType.NONE, expected=None, predicted=None),
            ErrorSpan(
                text="Music",
                error=ErrorType.FALSE_POSITIVE,
                expected=None,
                predicted="LOCATION",
            ),
            ErrorSpan(
                text=".",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
        ],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    assert create_prediction_error_span(parameter) == expected


def test_partial_match() -> None:

    parameter = LabelPredictionText(
        text="George Washington was a President in America, but that was long ago.",
        labels=[
            Label(text="George Washington", start=0, end=17, entity_type="PERSON"),
            Label(text="America", start=37, end=44, entity_type="LOCATION"),
        ],
        predictions=[
            Label(text="Washington", start=7, end=17, entity_type="LOCATION"),
            Label(text="President in America, but", start=24, end=49, entity_type="LOCATION"),
        ],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    expected = PredictionErrorSpans(
        spans=[
            ErrorSpan(text="George ", error=ErrorType.PARTIAL_FALSE_NEGATIVE, expected="PERSON", predicted=None),
            ErrorSpan(
                text="Washington",
                error=ErrorType.PARTIAL_TYPE_MISMATCH,
                expected="PERSON",
                predicted="LOCATION",
            ),
            ErrorSpan(
                text=" was a ",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
            ErrorSpan(
                text="President in ",
                error=ErrorType.PARTIAL_FALSE_POSITIVE,
                expected=None,
                predicted="LOCATION",
            ),
            ErrorSpan(
                text="America",
                error=ErrorType.PARTIAL_MATCH,
                expected="LOCATION",
                predicted="LOCATION",
            ),
            ErrorSpan(
                text=", but",
                error=ErrorType.PARTIAL_FALSE_POSITIVE,
                expected=None,
                predicted="LOCATION",
            ),
            ErrorSpan(
                text=" that was long ago.",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
        ],
        dataset_text_id=0,
        dataset_type=DatasetType.TRAIN,
    )

    actual = create_prediction_error_span(parameter)

    assert actual == expected


def test_partial_conll_validation_1143() -> None:

    parameter = LabelPredictionText(
        text="Economist Joel Naroff of First Union bank in Philadelphia said ...",
        labels=[
            Label(text="Joel Naroff", start=10, end=21, entity_type="PER"),
            Label(text="First Union", start=25, end=36, entity_type="ORG"),
            Label(text="Philadelphia", start=45, end=57, entity_type="LOC"),
        ],
        predictions=[
            Label(text="Joel Naroff", start=10, end=21, entity_type="PER"),
            Label(text="First Union bank", start=25, end=41, entity_type="ORG"),
            Label(text="Philadelphia", start=45, end=57, entity_type="LOC"),
        ],
        dataset_text_id=1143,
        dataset_type=DatasetType.VALIDATION,
    )

    expected = PredictionErrorSpans(
        spans=[
            ErrorSpan(text="Economist ", error=ErrorType.NONE, expected=None, predicted=None),
            ErrorSpan(
                text="Joel Naroff",
                error=ErrorType.MATCH,
                expected="PER",
                predicted="PER",
            ),
            ErrorSpan(
                text=" of ",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
            ErrorSpan(
                text="First Union",
                error=ErrorType.PARTIAL_MATCH,
                expected="ORG",
                predicted="ORG",
            ),
            ErrorSpan(
                text=" bank",
                error=ErrorType.PARTIAL_FALSE_POSITIVE,
                expected=None,
                predicted="ORG",
            ),
            ErrorSpan(
                text=" in ",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
            ErrorSpan(
                text="Philadelphia",
                error=ErrorType.MATCH,
                expected="LOC",
                predicted="LOC",
            ),
            ErrorSpan(
                text=" said ...",
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            ),
        ],
        dataset_text_id=1143,
        dataset_type=DatasetType.VALIDATION,
    )

    actual = create_prediction_error_span(parameter)

    assert actual == expected

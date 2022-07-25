import abc
import random
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import (
    DatasetType,
    ErrorSpan,
    ErrorType,
    Label,
    LabeledText,
    LabeledTokenizedText,
    LabelPredictionText,
    PredictionErrorSpans,
    PreTokenizedText,
    SectionType,
)
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.utils.dash import error_span_view, paginated_table

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


def __prediction_partial_match(
    text: str, label_idx: int, pred_idx: int, last_pos: int, preds: List[Label], labels: List[Label]
) -> Tuple[List[ErrorSpan], int, int, int]:
    spans: List[ErrorSpan] = []
    end_pred = preds[pred_idx].end
    end_label = labels[label_idx].end
    pred_n = len(preds)
    label_n = len(labels)

    while last_pos < max(end_pred, end_label):
        spans.append(
            ErrorSpan.construct(
                text=text[last_pos : min(end_pred, end_label)],
                error=ErrorType.PARTIAL_MATCH
                if labels[label_idx].entity_type == preds[pred_idx].entity_type
                else ErrorType.PARTIAL_TYPE_MISMATCH,
                expected=labels[label_idx].entity_type,
                predicted=preds[pred_idx].entity_type,
            )
        )
        last_pos = min(end_pred, end_label)
        if end_pred < end_label:
            pred_idx += 1
            if pred_idx < pred_n:
                if preds[pred_idx].start >= end_label:
                    spans.append(
                        ErrorSpan.construct(
                            text=text[last_pos:end_label],
                            error=ErrorType.PARTIAL_FALSE_NEGATIVE,
                            expected=labels[label_idx].entity_type,
                            predicted=None,
                        )
                    )
                    label_idx += 1
                    last_pos = end_label
                elif last_pos < preds[pred_idx].start:
                    end_pred = preds[pred_idx].end
                    spans.append(
                        ErrorSpan.construct(
                            text=text[last_pos : preds[pred_idx].start],
                            error=ErrorType.PARTIAL_FALSE_NEGATIVE,
                            expected=labels[label_idx].entity_type,
                            predicted=None,
                        )
                    )
                    last_pos = preds[pred_idx].start
            else:
                spans.append(
                    ErrorSpan.construct(
                        text=text[last_pos:end_label],
                        error=ErrorType.PARTIAL_FALSE_NEGATIVE,
                        expected=labels[label_idx].entity_type,
                        predicted=None,
                    )
                )
                label_idx += 1
                last_pos = end_label
        elif end_pred > end_label:
            label_idx += 1
            if label_idx < label_n:
                if labels[label_idx].start >= end_pred:
                    spans.append(
                        ErrorSpan.construct(
                            text=text[last_pos:end_pred],
                            error=ErrorType.PARTIAL_FALSE_POSITIVE,
                            expected=None,
                            predicted=preds[pred_idx].entity_type,
                        )
                    )
                    pred_idx += 1
                    last_pos = end_pred
                elif last_pos < labels[label_idx].start:
                    end_label = labels[label_idx].end
                    spans.append(
                        ErrorSpan.construct(
                            text=text[last_pos : labels[label_idx].start],
                            error=ErrorType.PARTIAL_FALSE_POSITIVE,
                            expected=None,
                            predicted=preds[pred_idx].entity_type,
                        )
                    )
                    last_pos = labels[label_idx].start
            else:
                spans.append(
                    ErrorSpan.construct(
                        text=text[last_pos:end_pred],
                        error=ErrorType.PARTIAL_FALSE_POSITIVE,
                        expected=None,
                        predicted=preds[pred_idx].entity_type,
                    )
                )
                pred_idx += 1
                last_pos = end_pred
        else:
            pred_idx += 1
            label_idx += 1
    return spans, label_idx, pred_idx, last_pos


def __prediction_error_spans(text: str, preds: List[Label], labels: List[Label]) -> Iterable[ErrorSpan]:
    last_pos = 0
    text_n = len(text)
    label_idx = 0
    pred_idx = 0
    label_n = len(labels)
    pred_n = len(preds)

    while last_pos < text_n:
        next_label = labels[label_idx].start if label_idx < label_n else text_n
        next_pred = preds[pred_idx].start if pred_idx < pred_n else text_n
        next_pos = min(next_pred, next_label)
        if last_pos < next_pos:
            yield ErrorSpan.construct(
                text=text[last_pos:next_pos],
                error=ErrorType.NONE,
                expected=None,
                predicted=None,
            )
            last_pos = next_pos
            if last_pos == text_n:
                break
        if next_pred < next_label and preds[pred_idx].end <= next_label:
            yield ErrorSpan.construct(
                text=preds[pred_idx].text,
                error=ErrorType.FALSE_POSITIVE,
                expected=None,
                predicted=preds[pred_idx].entity_type,
            )
            last_pos = preds[pred_idx].end
            pred_idx += 1
            continue
        if next_label < next_pred and labels[label_idx].end <= next_pred:
            yield ErrorSpan.construct(
                text=labels[label_idx].text,
                error=ErrorType.FALSE_NEGATIVE,
                expected=labels[label_idx].entity_type,
                predicted=None,
            )
            last_pos = labels[label_idx].end
            label_idx += 1
            continue
        if next_pred == next_label and preds[pred_idx].end == labels[label_idx].end:
            yield ErrorSpan.construct(
                text=labels[label_idx].text,
                error=ErrorType.MATCH
                if labels[label_idx].entity_type == preds[pred_idx].entity_type
                else ErrorType.TYPE_MISMATCH,
                expected=labels[label_idx].entity_type,
                predicted=preds[pred_idx].entity_type,
            )
            last_pos = labels[label_idx].end
            label_idx += 1
            pred_idx += 1
            continue
        if last_pos < max(next_pred, next_label):
            if next_pred < next_label:
                yield ErrorSpan.construct(
                    text=text[last_pos:next_label],
                    error=ErrorType.PARTIAL_FALSE_POSITIVE,
                    expected=None,
                    predicted=preds[pred_idx].entity_type,
                )
            else:
                yield ErrorSpan.construct(
                    text=text[last_pos:next_pred],
                    error=ErrorType.PARTIAL_FALSE_NEGATIVE,
                    expected=labels[label_idx].entity_type,
                    predicted=None,
                )
            last_pos = max(next_pred, next_label)
        partial_match_spans, label_idx, pred_idx, last_pos = __prediction_partial_match(
            text, label_idx, pred_idx, last_pos, preds, labels
        )
        yield from partial_match_spans


def create_prediction_error_span(prediction_label_text: LabelPredictionText) -> PredictionErrorSpans:
    return PredictionErrorSpans.construct(
        spans=list(
            __prediction_error_spans(
                prediction_label_text.text, prediction_label_text.predictions, prediction_label_text.labels
            )
        ),
        dataset_type=prediction_label_text.dataset_type,
        dataset_text_id=prediction_label_text.dataset_text_id,
    )


class PredictionErrorComponent(Component, abc.ABC):
    def __init__(self, examples: List[str]) -> None:
        self.examples = [PredictionErrorSpans.parse_raw(ex) for ex in examples]
        super(PredictionErrorComponent, self).__init__()

    @classmethod
    @abc.abstractmethod
    def get_tokenized_examples(cls, dataset: Dataset) -> Tuple[List[PreTokenizedText], List[LabeledText]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def table_caption(self) -> str:
        raise NotImplementedError()

    @classmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        predictable, labels = cls.get_tokenized_examples(dataset)
        predictions = predictor.predict(predictable)
        predictions_per_id: Dict[int, LabeledTokenizedText] = {pred.dataset_text_id: pred for pred in predictions}
        labels_per_id: Dict[int, LabeledText] = {label.dataset_text_id: label for label in labels}

        examples = [
            create_prediction_error_span(
                LabelPredictionText.from_prediction_label_pair(
                    predictions_per_id[text_id],
                    labels_per_id[text_id],
                )
            )
            for text_id in set(list(predictions_per_id.keys()) + list(labels_per_id.keys()))
            if predictions_per_id[text_id].labels != labels_per_id[text_id].labels
        ]
        random.shuffle(examples)

        return {"examples": [ex.json() for ex in examples]}

    def to_dash_components(self) -> List[DashComponent]:
        caption = self.table_caption()
        table, callback = paginated_table(
            self.component_name,
            [{"name": "Text Id", "id": "id"}, {"name": caption, "id": "ex"}],
            [
                {
                    "ex": error_span_view(self.component_name, example),
                    "id": f"{example.dataset_type}-{example.dataset_text_id}",
                }
                for example in self.examples
            ],
            caption=caption,
        )

        self._callbacks.append(callback)

        return table

    @property
    def section_type(self) -> SectionType:
        return SectionType.EXAMPLES


class TrainingExamplesComponent(PredictionErrorComponent):
    @classmethod
    def get_tokenized_examples(cls, dataset: Dataset) -> Tuple[List[PreTokenizedText], List[LabeledText]]:
        return dataset.train_tokenized, dataset.train

    def table_caption(self) -> str:
        return "Training Prediction Errors"

    dataset_requirements = (DatasetType.TRAIN,)
    component_name = "training-examples"


class ValidationExamplesComponent(PredictionErrorComponent):
    @classmethod
    def get_tokenized_examples(cls, dataset: Dataset) -> Tuple[List[PreTokenizedText], List[LabeledText]]:
        return dataset.val_tokenized, dataset.val

    def table_caption(self) -> str:
        return "Validation Prediction Errors"

    dataset_requirements = (DatasetType.VALIDATION,)
    component_name = "validation-examples"


class TestExamplesComponent(PredictionErrorComponent):
    @classmethod
    def get_tokenized_examples(cls, dataset: Dataset) -> Tuple[List[PreTokenizedText], List[LabeledText]]:
        return dataset.test_tokenized, dataset.test

    def table_caption(self) -> str:
        return "Test Prediction Errors"

    dataset_requirements = (DatasetType.TEST,)
    component_name = "test-examples"

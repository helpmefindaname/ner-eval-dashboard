from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from ner_eval_dashboard.datamodels import (
    ConfusionScores,
    Label,
    LabeledText,
    LabeledTokenizedText,
    TypeConfusionScores,
)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def acc(tp: int, n: int) -> float:
    if tp + n == 0:
        return 0
    return tp / (tp + n)


def acc_dict(tp: Dict[str, int], n: Dict[str, int], labels: List[str]) -> Dict[str, float]:
    return {label: acc(tp[label], n[label]) for label in labels}


def f1_score_dict(precisions: Dict[str, float], recalls: Dict[str, float], label_names: List[str]) -> Dict[str, float]:
    return {label: f1_score(recalls[label], precisions[label]) for label in label_names}


def list_to_type_span_dict(labels: Sequence[Label]) -> Dict[str, List[Tuple[int, int]]]:
    result: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for label in labels:
        result[label.entity_type].append((label.start, label.end))
    return result


def _iter_examples(
    predictions: List[LabeledTokenizedText], labels: List[LabeledText]
) -> Iterable[Tuple[Sequence[Label], Sequence[Label]]]:
    predictions_per_id: Dict[int, LabeledTokenizedText] = {pred.dataset_text_id: pred for pred in predictions}
    labels_per_id: Dict[int, LabeledText] = {label.dataset_text_id: label for label in labels}
    assert sorted(predictions_per_id.keys()) == sorted(labels_per_id.keys())

    for text_id in predictions_per_id.keys():
        text_labels = labels_per_id[text_id].labels
        text_predictions = predictions_per_id[text_id].labels
        yield text_labels, text_predictions


def compute_confusion_metrics_for_example(
    text_labels: Sequence[Label], text_predictions: Sequence[Label]
) -> ConfusionScores:
    scores = ConfusionScores()
    text_label_spans = {(label.start, label.end): label.entity_type for label in text_labels}
    text_prediction_spans = {(label.start, label.end): label.entity_type for label in text_predictions}
    for span in set(list(text_label_spans.keys()) + list(text_prediction_spans.keys())):
        if span not in text_label_spans:
            scores.overlap_fp += 1
            scores.micro_fp += 1
            scores.cls_fps[text_prediction_spans[span]] += 1
        if span not in text_prediction_spans:
            scores.overlap_fn += 1
            scores.micro_fn += 1
            scores.cls_fns[text_label_spans[span]] += 1
        if span in text_label_spans and span in text_prediction_spans:
            scores.overlap_tp += 1
            if text_prediction_spans[span] == text_label_spans[span]:
                scores.micro_tp += 1
                scores.cls_tps[text_prediction_spans[span]] += 1
            else:
                scores.micro_fp += 1
                scores.micro_fn += 1
                scores.cls_fps[text_prediction_spans[span]] += 1
                scores.cls_fns[text_label_spans[span]] += 1
    return scores


def compute_type_confusion_metrics_for_example(
    text_labels: Sequence[Label], text_predictions: Sequence[Label], label_names: List[str]
) -> TypeConfusionScores:
    scores = TypeConfusionScores()
    label_dict = list_to_type_span_dict(text_labels)
    predictions_dict = list_to_type_span_dict(text_predictions)
    for label_name in label_names:
        label_spans = label_dict[label_name]
        prediction_spans = predictions_dict[label_name]

        i = 0
        j = 0
        while i < len(label_spans) and j < len(prediction_spans):
            label_start, label_end = label_spans[i]
            pred_start, pred_end = prediction_spans[j]
            if label_start < pred_end and pred_start < label_end:
                scores.type_tps[label_name] += 1
                i += 1
                j += 1
                continue

            if label_start < pred_start:
                scores.type_fns[label_name] += 1
                i += 1
            else:
                scores.type_fps[label_name] += 1
                j += 1

        while i < len(label_spans):
            i += 1
            scores.type_fns[label_name] += 1
        while j < len(prediction_spans):
            j += 1
            scores.type_fps[label_name] += 1
    return scores


def compute_confusion_metrics_for_examples(
    predictions: List[LabeledTokenizedText], labels: List[LabeledText]
) -> ConfusionScores:
    confusion_scores = ConfusionScores()

    for text_labels, text_predictions in _iter_examples(predictions, labels):
        confusion_scores += compute_confusion_metrics_for_example(text_labels, text_predictions)
    return confusion_scores


def compute_type_confusion_metrics_for_examples(
    predictions: List[LabeledTokenizedText], labels: List[LabeledText], label_names: List[str]
) -> TypeConfusionScores:
    type_scores = TypeConfusionScores()

    for text_labels, text_predictions in _iter_examples(predictions, labels):
        type_scores += compute_type_confusion_metrics_for_example(text_labels, text_predictions, label_names)
    return type_scores

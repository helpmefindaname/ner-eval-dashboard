from typing import TYPE_CHECKING, Any, Dict, List

from dash import html
from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import (
    DatasetType,
    LabeledText,
    LabeledTokenizedText,
    SectionType,
)
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.utils.table import create_table_from_records

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def acc(tp: int, n: int) -> float:
    if tp + n == 0:
        return 0
    return tp / (tp + n)


def acc_dict(tp: dict[str, int], n: dict[str, int], labels: list[str]) -> Dict[str, float]:
    return {label: acc(tp[label], n[label]) for label in labels}


def f1_score_dict(precisions: dict[str, float], recalls: dict[str, float], label_names: List[str]) -> dict[str, float]:
    return {label: f1_score(recalls[label], precisions[label]) for label in label_names}


def with_name(values: dict[str, float], name: str) -> dict[str, Any]:
    return {**values, "name": name}


class F1MetricComponent(Component):
    dataset_requirements = (DatasetType.TEST,)
    component_name = "f1-metrics"

    def __init__(
        self,
        label_names: List[str],
        recalls: Dict[str, float],
        precisions: Dict[str, float],
        type_recalls: Dict[str, float],
        type_precisions: Dict[str, float],
        overlap_recall: float,
        overlap_precision: float,
        micro_recall: float,
        micro_precision: float,
    ):
        f1 = f1_score_dict(precisions, recalls, label_names)

        type_f1 = f1_score_dict(type_precisions, type_recalls, label_names)

        for d in [recalls, precisions, f1, type_recalls, type_precisions, type_f1]:
            d["avg"] = sum(d.values()) / len(d)

        self.detailed_table = [
            with_name(recalls, "Recall"),
            with_name(precisions, "Precision"),
            with_name(f1, "F1"),
            with_name(type_recalls, "Type-Recall"),
            with_name(type_precisions, "Type_precision"),
            with_name(type_f1, "Type-F1"),
        ]
        self.detailed_header = (
            [{"name": "Metric", "id": "name"}]
            + [{"name": label, "id": label, "format": "{:.2%}"} for label in label_names]
            + [{"name": "Avg.", "id": "avg", "format": "{:.2%}"}]
        )

        overlap_f1 = f1_score(overlap_precision, overlap_recall)
        micro_f1 = f1_score(micro_precision, micro_recall)

        self.simple_table = [
            {"name": name, "value": value}
            for name, value in [
                ("Macro-F1", f1["avg"]),
                ("Micro-F1", micro_f1),
                ("Overlap-F1", overlap_f1),
                ("Micro-Precision", micro_precision),
                ("Micro-Recall", micro_recall),
                ("Overlap-Precision", overlap_precision),
                ("Overlap-Recall", overlap_recall),
            ]
        ]
        self.simple_header = [
            {"name": "Metric", "id": "name"},
            {"name": "value", "id": "value", "format": "{:.2%}"},
        ]

        super(F1MetricComponent, self).__init__()

    @classmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        label_names = predictor.label_names
        micro_tp = 0
        micro_fp = 0
        micro_fn = 0
        overlap_tp = 0
        overlap_fp = 0
        overlap_fn = 0
        type_tps = {n: 0 for n in label_names}
        type_fps = {n: 0 for n in label_names}
        type_fns = {n: 0 for n in label_names}
        cls_tps = {n: 0 for n in label_names}
        cls_fps = {n: 0 for n in label_names}
        cls_fns = {n: 0 for n in label_names}

        predictions = predictor.predict(dataset.test_tokenized)
        predictions_per_id: Dict[int, LabeledTokenizedText] = {pred.dataset_text_id: pred for pred in predictions}
        labels_per_id: Dict[int, LabeledText] = {label.dataset_text_id: label for label in dataset.test}
        assert sorted(predictions_per_id.keys()) == sorted(labels_per_id.keys())

        for text_id in predictions_per_id.keys():
            text_labels = labels_per_id[text_id].labels
            text_predictions = predictions_per_id[text_id].labels

        return dict(
            label_names=label_names,
            recalls=acc_dict(cls_tps, cls_fns, label_names),
            precisions=acc_dict(cls_tps, cls_fps, label_names),
            type_recalls=acc_dict(type_tps, type_fns, label_names),
            type_precisions=acc_dict(type_tps, type_fps, label_names),
            overlap_recall=acc(overlap_tp, overlap_fn),
            overlap_precision=acc(overlap_tp, overlap_fp),
            micro_recall=acc(micro_tp, micro_fn),
            micro_precision=acc(micro_tp, micro_fp),
        )

    def to_dash_components(self) -> List[DashComponent]:
        return [
            html.Div(
                [
                    create_table_from_records(
                        self.simple_header,
                        self.simple_table,
                        caption="Evaluation metrics",
                    ),
                ],
                className="col-md-3 col-sm-3",
            ),
            html.Div(
                [
                    create_table_from_records(
                        self.detailed_header,
                        self.detailed_table,
                        caption="Detailed Precision Recall F1 scores",
                    ),
                ],
                className="col-md-9 col-sm-9",
            ),
        ]

    @property
    def section_type(self) -> SectionType:
        return SectionType.BASIC_METRICS

from typing import TYPE_CHECKING, Any, Dict, List

from dash import html
from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import DatasetType, SectionType
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.utils.dash import create_table_from_records
from ner_eval_dashboard.utils.metrics import (
    acc,
    acc_dict,
    compute_confusion_metrics_for_examples,
    compute_type_confusion_metrics_for_examples,
    f1_score,
    f1_score_dict,
)

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


def with_name(values: Dict[str, float], name: str) -> Dict[str, Any]:
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
            with_name(type_precisions, "Type-precision"),
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
                ("Boundary-F1", overlap_f1),
                ("Micro-Precision", micro_precision),
                ("Micro-Recall", micro_recall),
                ("Boundary-Precision", overlap_precision),
                ("Boundary-Recall", overlap_recall),
            ]
        ]
        self.simple_header = [
            {"name": "Metric", "id": "name"},
            {"name": "value", "id": "value", "format": "{:.2%}"},
        ]

        super().__init__()

    @classmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        label_names = predictor.label_names
        predictions = predictor.predict(dataset.test_tokenized)

        confusion_scores = compute_confusion_metrics_for_examples(predictions, dataset.test)
        type_scores = compute_type_confusion_metrics_for_examples(predictions, dataset.test, label_names)

        return dict(
            label_names=label_names,
            recalls=acc_dict(confusion_scores.cls_tps, confusion_scores.cls_fns, label_names),
            precisions=acc_dict(confusion_scores.cls_tps, confusion_scores.cls_fps, label_names),
            type_recalls=acc_dict(type_scores.type_tps, type_scores.type_fns, label_names),
            type_precisions=acc_dict(type_scores.type_tps, type_scores.type_fps, label_names),
            overlap_recall=acc(confusion_scores.overlap_tp, confusion_scores.overlap_fn),
            overlap_precision=acc(confusion_scores.overlap_tp, confusion_scores.overlap_fp),
            micro_recall=acc(confusion_scores.micro_tp, confusion_scores.micro_fn),
            micro_precision=acc(confusion_scores.micro_tp, confusion_scores.micro_fp),
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

from typing import TYPE_CHECKING, Any, Dict, List

import dash_bootstrap_components as dbc
from dash import dash_table
from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import SectionType, DatasetType
from ner_eval_dashboard.dataset import Dataset

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def acc(tp: int, n: int):
    if tp + n == 0:
        return 0
    return tp / (tp + n)


def acc_dict(tp: dict[str, int], n: dict[str, int], labels: list[str]):
    return {label: acc(tp[label], n[label]) for label in labels}


def f1_score_dict(precisions: dict[str, float], recalls: dict[str, float], label_names: List[str]) -> dict[str, float]:
    return {label: f1_score(recalls[label], precisions[label]) for label in label_names}


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
        recalls["name"] = "Recall"
        precisions["name"] = "Precision"
        f1["name"] = "F1"
        type_recalls["name"] = "Type-Recall"
        type_precisions["name"] = "Type-Precision"
        type_f1["name"] = "Type-F1"

        self.detailed_table = [recalls, precisions, f1, type_recalls, type_precisions, type_f1]
        self.detailed_header = (
            [{"name": "Metric", "id": "name"}]
            + [{"name": label, "id": label} for label in label_names]
            + [{"name": "Avg.", "id": "avg"}]
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
            {"name": "value", "id": "value"},
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

    def to_dash_component(self) -> DashComponent:
        return dbc.Container(
            [
                dbc.Label("Evaluation metrics"),
                dash_table.DataTable(
                    self.simple_table,
                    self.simple_header,
                ),
                dbc.Label("Detailed Precision Recall F1 scores"),
                dash_table.DataTable(
                    self.detailed_table,
                    self.detailed_header,
                ),
            ]
        )

    @property
    def section_type(self) -> SectionType:
        return SectionType.BASIC_METRICS

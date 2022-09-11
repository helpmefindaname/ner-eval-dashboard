import statistics
from math import sqrt
from typing import TYPE_CHECKING, Any, Dict, List

from dash import html
from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import DatasetType, SectionType
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.predictor_mixins import DropoutPredictorMixin
from ner_eval_dashboard.utils.dash import create_table_from_records
from ner_eval_dashboard.utils.metrics import (
    acc,
    acc_dict,
    compute_confusion_metrics_for_examples,
    f1_score,
    f1_score_dict,
)

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


def with_name(name: str, values: List[float]) -> Dict[str, Any]:
    m = statistics.mean(values)
    std = sqrt(statistics.variance(values, m))
    return {"name": name, "mean": m, "min": min(values), "max": max(values), "std-dev": std}


class DropoutRobustnessComponent(Component):
    dataset_requirements = (DatasetType.TEST,)

    component_name = "dropout_robustness"

    N_EVALS = 5

    def __init__(
        self,
        label_names: List[str],
        micro_precisions: List[float],
        micro_recalls: List[float],
        precisions: List[Dict[str, float]],
        recalls: List[Dict[str, float]],
    ):

        macro_f1s = [statistics.mean(f1_score_dict(p, r, label_names).values()) for p, r in zip(precisions, recalls)]
        micro_f1s = [f1_score(p, r) for p, r in zip(micro_precisions, micro_recalls)]

        self.table = [
            with_name("Macro-F1", macro_f1s),
            with_name("Micro-F1", micro_f1s),
        ]
        self.header = [
            {"name": "Metric", "id": "name"},
            {"name": "Mean", "id": "mean", "format": "{:.2%}"},
            {"name": "Standard-deviation", "id": "std-dev", "format": "{:.2%}"},
            {"name": "Min", "id": "min", "format": "{:.2%}"},
            {"name": "Max", "id": "max", "format": "{:.2%}"},
        ]

        super(DropoutRobustnessComponent, self).__init__()

    @classmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        label_names = predictor.label_names

        assert isinstance(predictor, DropoutPredictorMixin)

        micro_recalls: List[float] = []
        micro_precisions: List[float] = []
        precisions: List[Dict[str, float]] = []
        recalls: List[Dict[str, float]] = []

        for _ in range(cls.N_EVALS):
            predictions = predictor.predict_with_dropout(dataset.test_tokenized)

            confusion_scores = compute_confusion_metrics_for_examples(predictions, dataset.test)
            micro_precisions.append(acc(confusion_scores.micro_tp, confusion_scores.micro_fp))
            micro_recalls.append(acc(confusion_scores.micro_tp, confusion_scores.micro_fn))
            precisions.append(acc_dict(confusion_scores.cls_tps, confusion_scores.cls_fps, label_names))
            recalls.append(acc_dict(confusion_scores.cls_tps, confusion_scores.cls_fns, label_names))

        return dict(
            label_names=label_names,
            micro_precisions=micro_precisions,
            micro_recalls=micro_recalls,
            precisions=precisions,
            recalls=recalls,
        )

    def to_dash_components(self) -> List[DashComponent]:
        return [
            html.P(
                "Notice: this metrics depends on the amount of dropout used in the specific Predictor. "
                "Therefore it can heavily vary in different architectures."
            ),
            html.Div(
                [
                    create_table_from_records(
                        self.header,
                        self.table,
                        caption="Robustnes metrics using dropout as variation.",
                    ),
                ],
                className="col-md-12 col-sm-12",
            ),
        ]

    @property
    def section_type(self) -> SectionType:
        return SectionType.ROBUSTNESS

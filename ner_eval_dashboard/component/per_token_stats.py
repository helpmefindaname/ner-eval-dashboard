import math
import statistics
from collections import defaultdict
from math import sqrt
from typing import TYPE_CHECKING, Any, Dict, List

from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import DatasetType, SectionType
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.predictor_mixins import ScoredTokenPredictorMixin
from ner_eval_dashboard.utils.dash import paginated_table

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


def with_name(name: str, values: List[float]) -> Dict[str, Any]:
    m = statistics.mean(values)
    std = sqrt(statistics.variance(values, m))
    return {"name": name, "mean": m, "min": min(values), "max": max(values), "std-dev": std}


class PerTokenStats(Component):
    dataset_requirements = (DatasetType.TEST,)

    component_name = "per_token_stats"
    eps = 1e-10

    def __init__(
        self,
        tag_label_names: List[str],
        token_statistics: List[Dict[str, Any]],
        tag_statistics: List[Dict[str, Any]],
    ) -> None:
        super(PerTokenStats, self).__init__()
        self.tag_label_names = tag_label_names
        self.token_statistics = token_statistics
        self.tag_statistics = tag_statistics

        self.token_header = [
            {"name": "Token", "id": "token"},
            {"name": "Token count", "id": "count"},
            {"name": "Loss mean", "id": "mean", "format": "{:.2}"},
            {"name": "Loss Sum", "id": "sum", "format": "{:.2}"},
        ]
        self.tag_header = [
            {"name": "Tag", "id": "tag"},
            {"name": "Tag count", "id": "count"},
            {"name": "Loss mean", "id": "mean", "format": "{:.2}"},
            {"name": "Loss Sum", "id": "sum", "format": "{:.2}"},
        ]

    @classmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        assert isinstance(predictor, ScoredTokenPredictorMixin)
        tag_label_names = predictor.tag_label_names
        tag_format = predictor.token_format

        token_count: Dict[str, int] = defaultdict(int)
        token_loss_sum: Dict[str, float] = defaultdict(float)

        tag_count: Dict[str, int] = defaultdict(int)
        tag_loss_sum: Dict[str, float] = defaultdict(float)

        all_tag_labels = dataset.get_test_token_labeled(tag_format=tag_format)
        scored_examples = predictor.predict_token_scores(dataset.test_tokenized)
        for tag_labels, scores in zip(all_tag_labels, scored_examples):
            for labeled_token, scored_token in zip(tag_labels.tokens, scores.tokens):
                score = scored_token.get_score_by_tag(labeled_token.entity_type)
                loss = -math.log(max(score, cls.eps))
                token_count[labeled_token.text] += 1
                token_loss_sum[labeled_token.text] += loss
                tag_count[labeled_token.entity_type] += 1
                tag_loss_sum[labeled_token.entity_type] += loss

        return dict(
            tag_label_names=tag_label_names,
            token_statistics=sorted(
                [
                    {
                        "token": text,
                        "sum": token_loss_sum[text],
                        "count": token_count[text],
                        "mean": token_loss_sum[text] / token_count[text],
                    }
                    for text in token_count.keys()
                ],
                key=lambda x: x["mean"],
                reverse=True,
            ),
            tag_statistics=sorted(
                [
                    {
                        "tag": tag,
                        "sum": tag_loss_sum[tag],
                        "count": tag_count[tag],
                        "mean": tag_loss_sum[tag] / tag_count[tag],
                    }
                    for tag in tag_count.keys()
                ],
                key=lambda x: x["mean"],
                reverse=True,
            ),
        )

    def to_dash_components(self) -> List[DashComponent]:
        tag_table, tag_callback = paginated_table(
            f"{self.component_name}-tag",
            self.tag_header,
            self.tag_statistics,
            caption="Tag statistics",
        )
        token_table, token_callback = paginated_table(
            f"{self.component_name}-token",
            self.token_header,
            self.token_statistics,
            caption="Token statistics",
        )

        self._callbacks.append(tag_callback)
        self._callbacks.append(token_callback)

        return [
            tag_table,
            token_table,
        ]

    @property
    def section_type(self) -> SectionType:
        return SectionType.DETAILED_METRICS

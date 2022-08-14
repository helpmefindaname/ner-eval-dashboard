import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Union

import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.development.base_component import Component as DashComponent
from explainaboard.loaders.file_loader import FileLoaderReturn
from explainaboard.processors.named_entity_recognition import NERProcessor
from explainaboard.utils.cache_api import write_statistics_to_cache

from ner_eval_dashboard.cache import cache_root
from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import DatasetType, SectionType, TokenLabeledText
from ner_eval_dashboard.dataset import Dataset

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


class ExplainaboardComponent(Component):
    dataset_requirements = (
        DatasetType.TRAIN,
        DatasetType.TEST,
    )
    component_name = "explainaboard"

    def __init__(self, results: Dict[str, Any], **kwargs):
        self.value = results["overall"]["F1"]["value"]
        self.low_confidence_f1 = results["overall"]["F1"]["confidence_score_low"]
        self.high_confidence_f1 = results["overall"]["F1"]["confidence_score_high"]
        self.fine_grained = results["fine_grained"]

        super(ExplainaboardComponent, self).__init__()

    @classmethod
    def precompute_train_features(self, dataset: Dataset, processor: NERProcessor) -> Dict[str, Any]:
        tokens_sequences = []
        tags_sequences = []

        vocab: dict[str, int] = defaultdict(int)
        tag_vocab: dict[str, int] = defaultdict(int)
        for sample in dataset.train_token_labeled:
            tokens = [t.text for t in sample.tokens]
            tags = [t.entity_type for t in sample.tokens]

            for token, tag in zip(tokens, tags):
                vocab[token] += 1
                tag_vocab[tag] += 1

            tokens_sequences += tokens
            tags_sequences += tags

        # econ and efre dictionaries
        econ_dic, efre_dic = processor.get_econ_efre_dic(tokens_sequences, tags_sequences)
        # vocab_rank: the rank of each word based on its frequency
        sorted_dict = {key: rank for rank, key in enumerate(sorted(set(vocab.values()), reverse=True), 1)}
        vocab_rank = {k: sorted_dict[v] for k, v in vocab.items()}

        return {
            "efre_dic": efre_dic,
            "econ_dic": econ_dic,
            "vocab": vocab,
            "vocab_rank": vocab_rank,
        }

    @classmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        metadata = {
            "dataset_name": dataset.name,
            "sub_dataset_name": None,
            "split_name": "test",
            "source_language": None,
            "target_language": None,
            "reload_stat": True,
            "conf_value": 0.05,
            "system_details": None,
            "custom_features": None,
        }
        os.environ["EXPLAINABOARD_CACHE"] = str(cache_root / "explainaboard")
        predictions = predictor.predict(dataset.test_tokenized)
        token_predictions = [TokenLabeledText.from_labeled_tokenized_text(pred) for pred in predictions]
        token_labels = dataset.test_token_labeled
        sys_output = FileLoaderReturn(
            [
                {
                    "id": pred.id,
                    "tokens": [token.text for token in pred.tokens],
                    "true_tags": [token.entity_type for token in labeled.tokens],
                    "pred_tags": [token.entity_type for token in pred.tokens],
                }
                for pred, labeled in zip(token_predictions, token_labels)
            ]
        )
        processor = NERProcessor()
        train_statistics = cls.precompute_train_features(dataset, processor)
        write_statistics_to_cache(train_statistics, dataset.name, None)
        output = processor.process(metadata, sys_output)
        return output.to_dict()

    @staticmethod
    def interval_to_str(interval: List[Union[str, float]]) -> str:
        if isinstance(interval[0], str):
            if len(interval) == 1:
                return interval[0]
            return "[" + ",".join(interval) + "]"
        return "[" + ",".join(map(str, [round(f, 2) for f in interval])) + "]"

    def bucket_plot(self, description: str, values: List[Dict[str, Any]]) -> DashComponent:
        values = [v for v in values if v["n_samples"] > 0]
        intervals = [self.interval_to_str(v["bucket_interval"]) for v in values]
        performances = [v["performances"][0] for v in values]
        sample_counts = [v["n_samples"] for v in values]

        values = [perf["value"] for perf in performances]
        confs_low = [perf["confidence_score_low"] for perf in performances]
        confs_high = [perf["confidence_score_high"] for perf in performances]

        df = pd.DataFrame.from_dict(
            {
                "interval": intervals,
                "value": values,
                "conf_low": confs_low,
                "conf_high": confs_high,
                "sample_count": sample_counts,
            }
        )
        df.value = df.value.round(decimals=4)
        df.conf_low = df.conf_low.round(decimals=4)
        df.conf_high = df.conf_high.round(decimals=4)

        f = px.bar(
            df,
            x="interval",
            y="value",
            title=description,
            error_y=df["conf_high"] - df["value"],
            error_y_minus=df["value"] - df["conf_low"],
            hover_data=["interval", "value", "conf_low", "conf_high", "sample_count"],
            range_y=[0, 1],
        ).update_layout(hovermode="x")
        return html.Div(dcc.Graph(figure=f), className="col-md-3 col-sm-3")

    def to_dash_components(self) -> List[DashComponent]:
        components = []
        descriptions = {
            "sentence_length": "F1 by sentence length",
            "span_density": "F1 by the ratio between all entity tokens and sentence tokens",
            "span_tokens": "F1 by span length",
            "span_tag": "F1 by tag of the span",
            "span_capitalness": "F1 by whether the span is capitalized",
            "span_rel_pos": "F1 by relative position of span in the segment",
            "span_chars": "F1 by number of characters in the span",
            "num_oov": "F1 by the number of out-of-vocabulary words",
            "fre_rank": "F1 by span frequency in the training data",
            "span_econ": "F1 by span label consistency",
            "span_efre": "F1 by the average rank of each word based on its frequency in the training data",
        }
        components.append(
            self.bucket_plot(
                "Overall score with confidence intervall",
                [
                    {
                        "n_samples": 1,
                        "bucket_interval": ["overall"],
                        "performances": [
                            {
                                "value": self.value,
                                "confidence_score_low": self.low_confidence_f1,
                                "confidence_score_high": self.high_confidence_f1,
                            }
                        ],
                    }
                ],
            )
        )

        for k in self.fine_grained:
            components.append(self.bucket_plot(descriptions[k], self.fine_grained[k]))
        return components

    @property
    def section_type(self) -> SectionType:
        return SectionType.DETAILED_METRICS

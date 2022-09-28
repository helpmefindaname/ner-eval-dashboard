import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Union, cast

import nltk
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.development.base_component import Component as DashComponent
from textflint import Config
from textflint import Dataset as TextFlintDataset
from textflint import Engine
from textflint.input.model.flint_model.flint_model_ner import FlintModelNER
from textflint.report.analyzer import Analyzer
from textflint.report.report_generator import ReportGenerator

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import DatasetType, PreTokenizedText, SectionType, TokenLabeledText
from ner_eval_dashboard.dataset import Dataset

if TYPE_CHECKING:
    from ner_eval_dashboard.predictor import Predictor


class PredictorTextFlintWrapper(FlintModelNER):
    def __init__(self, predictor: "Predictor"):
        self.predictor = predictor
        super().__init__(self)

    def __call__(self, *inputs):
        breakpoint()
        pass

    def get_model_grad(self, *inputs):
        breakpoint()
        pass

    def unzip_samples(self, data_samples):
        breakpoint()
        pass

    def predict_tags_from_words(self, word_sequences: List[List[str]], batch_size: int = -1) -> List[List[str]]:
        pre_tokenized_texts = [
            PreTokenizedText.from_tokens(
                words,
                text_id=i,
            )
            for i, words in enumerate(word_sequences)
        ]
        labels = self.predictor.predict(pre_tokenized_texts)
        return [
            [token.entity_type for token in TokenLabeledText.from_labeled_tokenized_text(label).tokens]
            for label in labels
        ]


class TextFlintComponent(Component):
    dataset_requirements = (DatasetType.TEST,)
    component_name = "text_flint"

    def __init__(self, transformation: Dict[str, Any], subpopulation: Dict[str, Any]) -> None:
        super().__init__()
        self.transformation = transformation
        self.subpopulation = subpopulation
        self.linguist_radar = Analyzer.json_to_linguistic_radar({"transformation": self.transformation})
        self.sunburst_df, self.sunburst_settings = Analyzer.json_to_sunburst({"transformation": self.transformation})
        self.bar_df, self.bar_cols = Analyzer.json_to_bar_chart({
            "transformation": self.transformation,
            "subpopulation": self.subpopulation,
        })

    @staticmethod
    def init_nltk():
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle")
        except (LookupError, OSError):
            nltk.download("averaged_perceptron_tagger")

    @classmethod
    def precompute(cls, predictor: "Predictor", dataset: Dataset) -> Dict[str, Any]:
        cls.init_nltk()
        engine = Engine()
        text_flint_dataset = TextFlintDataset(task="NER")
        text_flint_dataset.extend(
            [
                {
                    "x": [token.text for token in example.tokens],
                    "y": [token.entity_type for token in example.tokens],
                }
                for example in dataset.get_test_token_labeled()
            ]
        )
        with tempfile.TemporaryDirectory() as f:
            text_flint_config = Config(task="NER", out_dir=f)
            text_flint_model = PredictorTextFlintWrapper(predictor)
            return engine.generate(text_flint_dataset, text_flint_config, text_flint_model)

    def to_dash_components(self) -> List[DashComponent]:

        radar_fig = ReportGenerator.get_radar_fig(self.linguist_radar)
        sunburst_fig = ReportGenerator.get_sunburst_fig(self.sunburst_df, self.sunburst_settings)
        bar_chart_fig = ReportGenerator.get_bar_chart(self.bar_df, self.bar_cols)

        components = [
            html.Div(dcc.Graph(figure=radar_fig), className="col-md-6 col-sm-6"),
            html.Div(dcc.Graph(figure=sunburst_fig), className="col-md-6 col-sm-6"),
            dcc.Graph(figure=bar_chart_fig),
        ]

        return components

    @property
    def section_type(self) -> SectionType:
        return SectionType.ROBUSTNESS

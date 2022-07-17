from typing import Dict, List

from dash import html
from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import SectionType
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.predictor import Predictor


class Section:
    def __init__(self, section_type: SectionType, name: str, description: str, sections: Dict[SectionType, "Section"]):
        self._section_type = section_type
        self.name = name
        self.description = description
        sections[self._section_type] = self

    def create_section(self, components: List[Component]) -> DashComponent:
        return html.Div(
            children=[
                html.H2(children=self.name),
                html.P(children=self.description),
                html.Div(children=[c.to_dash_component() for c in components]),
            ]
        )


def create_sections() -> Dict[SectionType, Section]:
    sections: Dict[SectionType, Section] = {}

    Section(SectionType.BASIC_METRICS, "Basic Metrics", "Standard NER Metrics.", sections)

    return sections


def create_base_layout(dash_sections: List[DashComponent], predictor: Predictor, dataset: Dataset) -> DashComponent:
    return html.Div(
        children=[
            html.H1(children=f"Ner Eval Dashboard for Model {predictor.name} on {dataset.name}"),
            html.Div(children=dash_sections),
        ]
    )

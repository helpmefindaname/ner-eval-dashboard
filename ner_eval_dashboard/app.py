from collections import defaultdict
from typing import Dict, List, Type

import dash_bootstrap_components as dbc
from dash import Dash
from dash.development.base_component import Component as DashComponent

from ner_eval_dashboard.component import Component
from ner_eval_dashboard.datamodels import SectionType
from ner_eval_dashboard.dataset import Dataset
from ner_eval_dashboard.predictor import Predictor
from ner_eval_dashboard.section import Section, create_base_layout, create_sections


def __filter_components(
    components: List[Type[Component]],
    dataset: Dataset,
    use_components: List[str] = None,
    exclude_components: List[str] = None,
) -> List[Type[Component]]:
    if use_components:
        _use_components = set(use_components)
        components = [c for c in components if c.component_name in _use_components]

    if exclude_components:
        _exclude_components = set(exclude_components)
        components = [c for c in components if c.component_name not in _exclude_components]

    components = [c for c in components if c.can_apply(dataset)]

    return components


def __group_components(components: List[Component]) -> Dict[SectionType, List[Component]]:
    grouped_components: Dict[SectionType, List[Component]] = defaultdict(list)
    for comp in components:
        grouped_components[comp.section_type].append(comp)
    return grouped_components


def __combine_groups_and_sections(
    grouped: Dict[SectionType, List[Component]], sections: Dict[SectionType, Section]
) -> List[DashComponent]:
    return [sections[section_type].create_section(comps) for section_type, comps in grouped.items()]


def create_app(
    name: str,
    predictor: Predictor,
    dataset: Dataset,
    use_components: List[str] = None,
    exclude_components: List[str] = None,
) -> Dash:
    app = Dash(
        name,
        title=f"Ner-Eval-Dashboard for {predictor.name} on {dataset.name}",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )

    component_cls = __filter_components(
        predictor.components,
        use_components=use_components,
        exclude_components=exclude_components,
        dataset=dataset,
    )
    created_components = [comp_cls.create(predictor, dataset) for comp_cls in component_cls]

    grouped = __group_components(created_components)
    sections = create_sections()

    dash_sections = __combine_groups_and_sections(grouped, sections)

    app.layout = create_base_layout(dash_sections, predictor, dataset)

    return app

from typing import Any, Dict, List, Union

import dash_bootstrap_components as dbc
from dash import html
from dash.development.base_component import Component

from ner_eval_dashboard.datamodels import ErrorSpan, ErrorType, PredictionErrorSpans


def create_table_from_records(header: List[Dict[str, Any]], content: List[Dict[str, Any]], caption: str) -> dbc.Table:
    return dbc.Table(
        [
            html.Caption(caption),
            html.Thead([html.Tr(children=[html.Th(h["name"]) for h in header])]),
            html.Tbody(
                [
                    html.Tr(children=[html.Td(h.get("format", "{}").format(row[h["id"]])) for h in header])
                    for row in content
                ]
            ),
        ]
    )


def error_span_view(component_id: str, error_span: PredictionErrorSpans) -> Component:
    def span_to_component(idx: int, span: ErrorSpan) -> List[Union[str, Component]]:
        if span.error == ErrorType.NONE:
            return [span.text]

        if span.error == ErrorType.MATCH:
            tooltip_text = f"Correctly classified as {span.expected}"
        elif span.error == ErrorType.TYPE_MISMATCH:
            tooltip_text = f"Expected {span.expected} but got {span.predicted}"
        elif span.error == ErrorType.FALSE_POSITIVE:
            tooltip_text = f"Predicted {span.predicted} but expected nothing"
        elif span.error == ErrorType.FALSE_NEGATIVE:
            tooltip_text = f"Expected {span.expected} but predicted nothing"
        elif span.error == ErrorType.PARTIAL_MATCH:
            tooltip_text = f"Part was correctly classified as {span.expected}"
        elif span.error == ErrorType.PARTIAL_TYPE_MISMATCH:
            tooltip_text = f"Part expected {span.expected} but predicted {span.predicted}"
        elif span.error == ErrorType.PARTIAL_FALSE_POSITIVE:
            tooltip_text = f"Part predicted {span.predicted} but expected nothing"
        else:
            assert span.error == ErrorType.PARTIAL_FALSE_NEGATIVE
            tooltip_text = f"Part expected {span.expected} but predicted nothing"
        error_class_name = span.error.name.lower().replace("_", "-")
        span_id = f"{component_id}-{idx}-{error_span.dataset_type.name.lower()}-{error_span.dataset_text_id}"
        return [
            html.Mark(span.text, className=f"error-span {error_class_name}", id=span_id),
            dbc.Tooltip(tooltip_text, class_name=error_class_name, target=span_id, placement="top"),
        ]

    return html.P(children=[comp for i, span in enumerate(error_span.spans) for comp in span_to_component(i, span)])

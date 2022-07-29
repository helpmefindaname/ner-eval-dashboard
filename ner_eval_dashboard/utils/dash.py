from typing import Any, Dict, List, Optional, Tuple, Union

import dash_bootstrap_components as dbc
from dash import Input, Output, html
from dash.development.base_component import Component

from ner_eval_dashboard.datamodels import (
    Callback,
    ErrorSpan,
    ErrorType,
    LabeledText,
    LabeledTokenizedText,
    PredictionErrorSpans,
)
from ner_eval_dashboard.utils.constants import MAX_CLASS_COLORS


def __format_component(_header: Dict[str, Any], row: Dict[str, Any]) -> html.Td:
    element = row[_header["id"]]
    if isinstance(element, html.Td):
        return element
    if isinstance(element, Component) or (isinstance(element, list) and isinstance(element[0], Component)):
        return html.Td(element)
    return html.Td(_header.get("format", "{}").format(element))


def create_table_from_records(header: List[Dict[str, Any]], content: List[Dict[str, Any]], caption: str) -> dbc.Table:
    return dbc.Table(
        [
            html.Caption(caption),
            html.Thead([html.Tr(children=[html.Th(h["name"]) for h in header])]),
            html.Tbody([html.Tr(children=[__format_component(h, row) for h in header]) for row in content]),
        ]
    )


def paginated_table(
    component_id: str, header: List[Dict[str, Any]], content: List[Dict[str, Any]], caption: str, page_size: int = 10
) -> Tuple[Component, Callback]:
    rows = [html.Tr(children=[__format_component(h, row) for h in header]) for row in content]
    page_count = (len(rows) + page_size - 1) // page_size
    pagination_id = f"{component_id}-table-pagination"
    content_id = f"{component_id}-table-body"

    def update_page(page: Optional[int]) -> List[html.Tr]:
        if page is None:
            return rows[:page_size]
        page -= 1
        return rows[page * page_size : page * page_size + page_size]

    return html.Div(
        [
            dbc.Table(
                [
                    html.Caption(caption),
                    html.Thead([html.Tr(children=[html.Th(h["name"]) for h in header])]),
                    html.Tbody(rows[:page_size], id=content_id),
                ]
            ),
            dbc.Pagination(
                id=pagination_id,
                first_last=True,
                max_value=page_count,
                fully_expanded=False,
                previous_next=True,
            ),
        ]
    ), Callback([Input(pagination_id, "active_page")], Output(content_id, "children"), update_page)


def prediction_view(component_id: str, prediction_text: LabeledTokenizedText, labels: List[str]) -> Component:
    spans: List[Union[str, Component]] = []

    labeled_text = LabeledText.from_labeled_tokenized_text(prediction_text)
    last = 0
    for idx, label in enumerate(labeled_text.labels):
        if last < label.start:
            spans.append(labeled_text.text[last : label.start])

        entity_type_id = labels.index(label.entity_type) % MAX_CLASS_COLORS

        prediction_class_name = f"prediction-{entity_type_id}"
        prediction_id = f"{component_id}-prediction-{idx}-{prediction_text.id}"
        tooltip_text = label.entity_type

        spans.append(html.Mark(label.text, className=f"prediction-span {prediction_class_name}", id=prediction_id))
        spans.append(dbc.Tooltip(tooltip_text, class_name=prediction_class_name, target=prediction_id, placement="top"))

    if last < len(labeled_text.text):
        spans.append(labeled_text.text[last:])

    return html.P(children=spans)


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
        span_id = f"{component_id}-{idx}-{error_span.id}"
        return [
            html.Mark(span.text, className=f"error-span {error_class_name}", id=span_id),
            dbc.Tooltip(tooltip_text, class_name=error_class_name, target=span_id, placement="top"),
        ]

    return html.P(children=[comp for i, span in enumerate(error_span.spans) for comp in span_to_component(i, span)])

from typing import Any, Dict, List

import dash_bootstrap_components as dbc
from dash import html


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

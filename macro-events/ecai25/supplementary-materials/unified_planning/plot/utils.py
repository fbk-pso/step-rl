# Copyright 2021-2023 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import unified_planning as up
from unified_planning.model import FNode
from unified_planning.plans.plan import ActionInstance
from unified_planning.plans.stn_plan import STNPlanNode
from unified_planning.plans.contingent_plan import ContingentPlanNode
import networkx as nx
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
    Callable,
)

# Defaults
FIGSIZE = (13.0, 8.0)
ARROWSIZE = 20
MIN_NODE_SIZE = 4000
NODE_COLOR = "#1f78b4"
EDGE_COLOR = "k"
FONT_SIZE = 10
FONT_COLOR = "k"
EDGE_FONT_SIZE = 8
EDGE_FONT_COLOR = "k"
FIGSIZE_SCALE_FACTOR = 65  # A scale factor from the figure size of plotly vs matplotlib


def draw_base_graph(
    graph: nx.DiGraph,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    top_bottom: bool = False,
    generate_node_label: Optional[
        Union[
            Callable[["ContingentPlanNode"], str],
            Callable[["ActionInstance"], str],
            Optional[Callable[["STNPlanNode"], str]],
            Optional[Callable[[FNode], str]],
        ]
    ] = None,
    arrowsize: int = ARROWSIZE,
    node_size: Optional[Union[float, Sequence[float]]] = None,
    node_color: Union[str, Sequence[str]] = NODE_COLOR,
    edge_color: Union[str, Sequence[str]] = EDGE_COLOR,
    font_size: int = FONT_SIZE,
    font_color: str = FONT_COLOR,
    draw_networkx_kwargs: Optional[Dict[str, Any]] = None,
    prog: str = "dot",
):
    import matplotlib.pyplot as plt  # type: ignore[import]

    # input "sanitization"
    if generate_node_label is None:
        node_label: Callable[[Any], str] = str
    else:
        node_label = generate_node_label
    if draw_networkx_kwargs is None:
        draw_networkx_kwargs = {}

    # drawing part
    labels: Dict[Any, str] = dict(map(lambda x: (x, node_label(x)), graph.nodes))
    if node_size is None:
        font_factor = font_size * font_size * 10.7

        def length_factor(label_length: int) -> float:
            return label_length * label_length / 28

        node_size = [
            max(length_factor(max(len(labels[node]), 3)) * font_factor, MIN_NODE_SIZE)
            for node in graph.nodes
        ]
    if figsize is None:
        figsize = FIGSIZE
    assert figsize is not None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    pos = _generate_positions(graph, prog=prog, top_bottom=top_bottom)

    nx.draw_networkx(
        graph,
        pos,
        labels=labels,
        arrowstyle="-|>",
        arrowsize=arrowsize,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color,
        font_size=font_size,
        font_color=font_color,
        font_family="monospace",
        ax=ax,
        **draw_networkx_kwargs,
    )
    return fig, ax, pos


def _generate_positions(
    graph: nx.DiGraph, prog: str, top_bottom: bool
) -> Dict[Any, Tuple[float, float]]:
    """
    This method generates the position using the nx.nx_agraph.graphviz_layout
    method. It needs a wrapper to get different positions of different elements
    with the same string representation. In the method they are collapsed in the
    same position.
    """
    new_graph = nx.DiGraph()
    id_to_node: Dict[Any, Any] = {i: node for i, node in enumerate(graph.nodes)}
    node_to_id = {v: k for k, v in id_to_node.items()}

    for node, nbrdict in graph.adjacency():
        new_node = node_to_id[node]
        new_graph.add_node(new_node)
        for neighbour in nbrdict:
            new_graph.add_edge(new_node, node_to_id[neighbour])

    if top_bottom:
        new_graph.graph.setdefault("graph", {})["rankdir"] = "TB"
    else:
        new_graph.graph.setdefault("graph", {})["rankdir"] = "LR"

    new_pos = nx.nx_agraph.graphviz_layout(new_graph, prog=prog)

    pos = {id_to_node[i]: value for i, value in new_pos.items()}

    return pos

import networkx as nx
from .utils import get_k_order_neighbors, create_attr_dictionary
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display


def get_graphviz_layout(graph, reverse=False):
    if reverse:
        return nx.drawing.nx_agraph.graphviz_layout(graph.reverse(),
                                                    prog='dot',
                                                    args='-Grankdir=BT')
    else:
        return nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')


def draw(graph,
         # figure options:
         ax=None,
         figsize=(8, 6),
         reverse=False,
         # node subsetting options:
         node_subset=None,
         radius=1,
         include_shaded=True,
         # node drawing properties:
         node_color=None,
         node_shape=None,
         default_node_shape='s',
         default_node_color='#2b8cbe',
         # edge drawing properties:
         edge_color=None,
         edge_style=None,
         arrows=True,
         default_edge_color='#000000',
         default_edge_style='solid',
         # label options:
         labels=True,
         label_dict=None,
         font_color='white',
         font_size=8,
         **kwargs):
    """
    General purpose function for drawing genealogical objects
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if include_shaded:
        radius += 1

    # if drawing only a subset, create a sub-graph:
    if node_subset is not None:
        node_subset = get_k_order_neighbors(graph, node_subset, radius=radius)
        nodes = node_subset.keys()
        graph = nx.subgraph(graph, nodes)
    else:
        nodes = list(graph.nodes)

    # edges:
    edges = list(graph.edges)

    # -------------------------------------------------------
    # Filling in node and edge attributes:

    # Assign edge color:
    edge_col = create_attr_dictionary(edges, edge_color, default_edge_color)
    # Assign edge style:
    edge_sty = create_attr_dictionary(edges, edge_style, default_edge_style, invert=True)

    # Assign node color:
    node_col = create_attr_dictionary(nodes, node_color, default_node_color)
    # Assign node shape
    node_shp = create_attr_dictionary(nodes, node_shape, default_node_shape, invert=True)

    if node_subset is not None and include_shaded:
        for n in node_col.keys():
            if node_subset[n] == radius:
                node_col[n] = '#E9E9E9'

        for n1, n2 in edge_col.keys():
            if node_subset[n1] == radius or node_subset[n2] == radius:
                edge_col[(n1, n2)] = '#E9E9E9'

    # -------------------------------------------------------
    # Configure layout:

    pos = get_graphviz_layout(graph, reverse)

    if reverse:
        graph = graph.reverse()

    # -------------------------------------------------------
    # Draw nodes:
    for n_shp, node_li in node_shp.items():
        nx.draw_networkx_nodes(graph, pos, nodelist=node_li,
                               node_shape=n_shp,
                               node_color=[node_col[n] for n in node_li],
                               ax=ax, **kwargs)

    # -------------------------------------------------------
    # Draw edges:

    for e_sty, edge_li in edge_sty.items():
        nx.draw_networkx_edges(graph, pos, ax=ax,
                               edgelist=edge_li,
                               edge_color=[edge_col[e] for e in edge_li],
                               style=e_sty, arrows=arrows, **kwargs)

    # -------------------------------------------------------
    # Draw labels
    if labels:
        nx.draw_networkx_labels(graph, pos,
                                labels=label_dict,
                                font_color=font_color,
                                font_size=font_size, ax=ax)

    # Turn off axis
    ax.set_axis_off()

    return ax


def draw_aligner_path(hap, path_color='orange', **kwargs):
    pass



def draw_aligner(ped,
                 ts,
                 anchor_links,
                 figsize=(16, 8),
                 anchor_color_map='nipy_spectral',
                 ped_kwargs=None,
                 ts_kwargs=None):

    if ped_kwargs is None:
        ped_kwargs = {}
    if ts_kwargs is None:
        ts_kwargs = {}

    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    ped.draw(ax=axes[0], **ped_kwargs)
    ts.draw(ax=axes[1], **ts_kwargs)

    ped_layout = get_graphviz_layout(ped.graph)
    ts_layout = get_graphviz_layout(ts.graph, reverse=True)

    # Transform figure:
    ax0tr = axes[0].transData
    ax1tr = axes[1].transData
    figtr = fig.transFigure.inverted()

    cmap = matplotlib.cm.get_cmap(anchor_color_map)

    for i, (ts_n, ped_n) in enumerate(anchor_links.items()):
        ptB = figtr.transform(ax0tr.transform(ped_layout[ped_n]))
        ptE = figtr.transform(ax1tr.transform(ts_layout[ts_n]))

        arrow = matplotlib.patches.FancyArrowPatch(
            ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
            color=cmap(i / len(anchor_links)), connectionstyle="arc3,rad=0.3",
            arrowstyle='-', alpha=0.3,
            shrinkA=10, shrinkB=10, linestyle='dashed'
        )

        fig.patches.append(arrow)

    return fig


def draw_interactive(genea_obj, **kwargs):
    """
    This method facilitates drawing interactive genealogical objects
    by providing an interface for `focused view`, where we may zoom in
    on a subset of the nodes only.

    :param genea_obj:
    :return:
    """

    ns_text = widgets.Text(
        value='',
        placeholder='List of nodes (comma separated)',
        description='Node subset:',
        disabled=False
    )

    rad_text = widgets.Text(
        value='1',
        placeholder='1',
        description='Radius',
        disabled=False
    )

    btn = widgets.Button(description="Plot")
    output = widgets.Output()

    def callback(wdgt):

        ns = []

        for n in ns_text.value.split(","):
            try:
                n = int(n.strip())
            except ValueError:
                continue

            if n in genea_obj.nodes:
                ns.append(n)

        if len(ns) < 1:
            ns = None

        radius = int(rad_text.value.strip())
        if radius < 1:
            radius = 1

        with output:
            output.clear_output()
            genea_obj.draw(node_subset=ns, radius=radius, **kwargs)
            plt.show()

    btn.on_click(callback)

    display(ns_text, rad_text, btn, output)
    btn.click()

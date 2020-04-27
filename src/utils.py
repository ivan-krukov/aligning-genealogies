import networkx as nx
import pandas as pd


def all_leaves_under_nodes(gg, nodes=None):

    if nodes is None:
        nodes = list(gg.nodes())

    ntl = {}  # Nodes to leaves

    for n in nodes:

        ntl[n] = set()
        n_children = list(gg.successors(n))

        if len(n_children) == 0:
            ntl[n].add(n)
        else:
            while len(n_children) > 0:
                nn, nn_children = n_children[0], list(gg.successors(n_children[0]))

                if len(nn_children) > 0:
                    n_children.extend(nn_children)
                else:
                    ntl[n].add(nn)

                del n_children[0]

    return ntl


def get_mrca(gg, pairs=None):
    return nx.all_pairs_lowest_common_ancestor(gg, pairs)


def convert_msprime_genealogy_to_nx(fname, directed=False):

    gen_df = pd.read_csv(fname, sep="\t",
                         names=["ind_id", "father", "mother", "time"])

    gg = [nx.Graph(), nx.DiGraph()][directed]

    # -------------------------------------------------------
    # Add all nodes and edges to the graph:
    gg.add_edges_from(zip(gen_df["father"], gen_df["ind_id"]))
    gg.add_edges_from(zip(gen_df["mother"], gen_df["ind_id"]))

    # -------------------------------------------------------
    # Add node attributes:
    nx.set_node_attributes(gg, dict(zip(gen_df['ind_id'], gen_df['time'])), 'time')

    def infer_sex(node_id):
        if node_id in gen_df['father']:
            return 'M'
        elif node_id in gen_df['mother']:
            return 'F'
        else:
            return 'U'

    gen_df['sex'] = gen_df['ind_id'].apply(infer_sex)

    nx.set_node_attributes(gg, dict(zip(gen_df['ind_id'], gen_df['sex'])), 'sex')

    return gg

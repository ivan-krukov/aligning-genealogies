from genealogy_aligner import Pedigree
from genealogy_aligner.kinship import kinship_matrix, kinship_traversal
from genealogy_aligner.utils import integer_dict
import pandas as pd
import numpy as np
from itertools import repeat, count
import networkx as nx
import pytest


def get_test_pedigree(name):
    return Pedigree.read_balsac("data/test/" + name + ".tsv")


def get_test_pedigree_table(name):
    return pd.read_table("data/test/" + name + ".tsv")


@pytest.mark.parametrize(
    "pedigree,founders",
    [
        ("simple", [1, 2, 3, 6]),
        ("disconnected", [1, 2, 3, 4, 5, 7]),
        ("multiple_founder", [1, 2, 3, 4, 6, 7]),
        ("proband_different_generations", [1, 2, 4, 5]),
        ("loop", [1, 2, 3, 6, 10]),
        ("inbreeding", [1, 2]),
        ("intergenerational", [1,2,3,4])
    ],
)
def test_founders(pedigree, founders):
    assert founders == get_test_pedigree(pedigree).founders()


@pytest.mark.parametrize(
    "pedigree,probands",
    [
        ("simple", [7, 8]),
        ("disconnected", [9, 10]),
        ("multiple_founder", [11, 12]),
        ("proband_different_generations", [7, 8]),
        ("loop", [11, 12]),
        ("inbreeding", [5,6]),
        ("intergenerational", [7,8])
    ],
)
def test_probands(pedigree, probands):
    assert probands == get_test_pedigree(pedigree).probands()


@pytest.mark.parametrize(
    "pedigree,bfs_fwd",
    [("simple", set([(1, 4), (1, 5), (2, 4), (2, 5), (3, 7), (4, 7), (5, 8), (6, 8)]))],
)
def test_iter_edges_forward(pedigree, bfs_fwd):
    assert bfs_fwd == set(get_test_pedigree(pedigree).iter_edges())


@pytest.mark.parametrize(
    "pedigree,bfs_bwd",
    [("simple", set([(7, 3), (7, 4), (8, 5), (8, 6), (4, 1), (4, 2), (5, 1), (5, 2)]))],
)
def test_iter_edges_backward(pedigree, bfs_bwd):
    assert bfs_bwd == set(get_test_pedigree(pedigree).iter_edges(forward=False))


@pytest.mark.parametrize(
    "pedigree,depth",
    [
        ("simple", [0, 0, 0, 1, 1, 0, 2, 2]),
        ("disconnected", [0, 0, 0, 0, 0, 1, 0, 1, 2, 2]),
        ("multiple_founder", [0, 0, 0, 0, 1, 0, 0, 1, 2, 2, 3, 3]),
        ("proband_different_generations", [0, 0, 1, 0, 0, 2, 2, 3]),
        ("loop", [0, 0, 0, 1, 1, 0, 2, 2, 3, 0, 3, 4]),
        ("inbreeding", [0,0,1,1,2,2]),
        ("intergenerational", [0,0,0,0,1,1,2,2])
    ],
)
def test_depth(pedigree, depth):
    assert integer_dict(depth) == get_test_pedigree(pedigree).infer_depth()


@pytest.mark.parametrize(
    "pedigree,depth",
    [
        ("simple", [2, 2, 1, 1, 1, 1, 0, 0]),
        ("multiple_founder", [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 0, 0]),
        ("proband_different_generations", [3, 3, 2, 2, 1, 1, 0, 0]),
        ("loop", [4, 4, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0]),
        ("inbreeding", [2,2,1,1,0,0]),
        ("intergenerational", [2,2,2,2,1,1,0,0])
    ],
)
def test_depth_backward(pedigree, depth):
    assert integer_dict(depth) == get_test_pedigree(pedigree).infer_depth(forward=False)


@pytest.mark.parametrize(
    "pedigree",
    [
        "simple",
        "disconnected",
        "multiple_founder",
        "multiple_marriages",
        "proband_different_generations",
        "loop",
        "loop_2",
        "loop_3",
        "inbreeding",
        "intergenerational"
    ],
)
def test_depth_ordering(pedigree):
    label = count(1)
    ped = get_test_pedigree(pedigree)
    depth = ped.infer_depth()
    ordered_nodes = sorted(ped.nodes, key=lambda n: depth[n])
    ordered_labels = [next(label) for n in ordered_nodes]
    relabelled = Pedigree(nx.relabel_nodes(ped.graph, dict(zip(ordered_nodes, ordered_labels))))
        
    for parent, child in relabelled.iter_edges():
        assert parent < child


@pytest.mark.parametrize(
    "pedigree",
    [
        "simple",
        "disconnected",
        "multiple_founder",
        "multiple_marriages",
        "proband_different_generations",
        "loop",
        "loop_2",
        "loop_3",
        "inbreeding",
        "intergenerational"
    ],
)
def test_depth_ordering_with_shuffle(pedigree):
    np.random.seed(100)
    label = count(1)
    ped = get_test_pedigree(pedigree)
    
    # shuffle labels
    orig_labels = list(ped.nodes)
    random_labels = np.random.choice(orig_labels, ped.n_individuals, replace=False)
    shuffled = Pedigree(nx.relabel_nodes(ped.graph, dict(zip(orig_labels, random_labels))))
    
    # infer and order by depth
    depth = shuffled.infer_depth()
    ordered_nodes = sorted(shuffled.nodes, key=lambda n: depth[n])
    ordered_labels = [next(label) for n in ordered_nodes]
    ordered = Pedigree(nx.relabel_nodes(shuffled.graph, dict(zip(ordered_nodes, ordered_labels))))
    
    # check oredring is correct
    for parent, child in ordered.iter_edges():
        assert parent < child
    


@pytest.mark.parametrize(
    "pedigree",
    [
        "simple",
        "disconnected",
        "multiple_founder",
        "proband_different_generations",
        "loop",
    ],
)
def test_sex(pedigree):
    ped_df = get_test_pedigree_table(pedigree)
    ped = get_test_pedigree(pedigree)
    inferred_sex = Pedigree.infer_sex(
        ped_df.individual.values, ped_df.father.values, ped_df.mother.values
    )

    expected_sex = ped.get_node_attr("sex")
    # we can't infer the sex of probands
    mask = dict(zip(ped.probands(), repeat(-1)))
    expected_sex.update(mask)
    assert integer_dict(inferred_sex) == expected_sex


@pytest.mark.parametrize(
    "pedigree",
    ["simple", "disconnected", "multiple_founder", "proband_different_generations", "loop", "loop_2", "loop_3"],
)
def test_kinship_calculation(pedigree):
    ped_df = get_test_pedigree_table(pedigree)
    ped = get_test_pedigree(pedigree)
    depth = ped.infer_depth()
    l = ped.n_individuals
    darray = np.array([depth[i] for i in range(1, l + 1)])

    K_genlib = kinship_matrix(ped_df.individual, ped_df.mother, ped_df.father, darray)
    K_traversal = ped.kinship_traversal(progress=False).todense()[1:, 1:]

    print(K_genlib - K_traversal)
    print(K_traversal)
    # use np.triu to get the upper triangle if we only store half the matrix
    assert np.allclose(K_genlib, K_traversal)
@pytest.mark.parametrize(
    "pedigree",
    ["inbreeding"]
)
def test_kinship_inbreeding(pedigree):
    ped = get_test_pedigree(pedigree)
    K_traversal = ped.kinship_traversal(progress=False)
    
    exact_inbreeding = np.array([
        [1/2, 0,   1/4, 1/4, 1/4, 1/4],
        [0,   1/2, 1/4, 1/4, 1/4, 1/4],
        [1/4, 1/4, 1/2, 1/4, 3/8, 3/8],
        [1/4, 1/4, 1/4, 1/2, 3/8, 3/8],
        [1/4, 1/4, 3/8, 3/8, 5/8, 3/8],
        [1/4, 1/4, 3/8, 3/8, 3/8, 5/8]
    ])
    ped_df = get_test_pedigree_table(pedigree)
    depth = ped.infer_depth()
    l = ped.n_individuals
    darray = np.array([depth[i] for i in range(1, l + 1)])
    K_genlib = kinship_matrix(ped_df.individual, ped_df.mother, ped_df.father, darray)

    assert np.allclose(exact_inbreeding, K_traversal.todense[1:, 1:])
    assert np.allclose(exact_inbreeding, K_genlib)
    

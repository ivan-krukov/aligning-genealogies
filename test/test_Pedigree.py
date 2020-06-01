from genealogy_aligner import Pedigree
from genealogy_aligner.utils import integer_dict
import pandas as pd
from itertools import repeat
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
    ],
)
def test_founders(pedigree, founders):
    assert founders == get_test_pedigree(pedigree).founders()


@pytest.mark.parametrize(
    "pedigree,probands",
    [("simple", [7, 8]), ("disconnected", [9, 10]), ("multiple_founder", [11, 12]),],
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
    ],
)
def test_depth(pedigree, depth):
    assert integer_dict(depth) == get_test_pedigree(pedigree).infer_depth()


@pytest.mark.parametrize(
    "pedigree,depth",
    [
        ("simple", [2, 2, 1, 1, 1, 1, 0, 0]),
        ("multiple_founder", [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 0, 0]),
    ],
)
def test_depth_backward(pedigree, depth):
    assert integer_dict(depth) == get_test_pedigree(pedigree).infer_depth(forward=False)

    
@pytest.mark.parametrize('pedigree', ['simple', 'disconnected', 'multiple_founder'])
def test_sex(pedigree):
    ped_df = get_test_pedigree_table(pedigree)
    
    ped = get_test_pedigree(pedigree)
    inferred_sex = Pedigree.infer_sex(ped_df.individual.values, ped_df.father.values, ped_df.mother.values)
    
    expected_sex = ped.get_node_attr('sex')
    # we can't infer the sex of probands
    mask = dict(zip(ped.probands(), repeat(-1)))
    expected_sex.update(mask)
    assert integer_dict(inferred_sex) == expected_sex
    

from genealogy_aligner import Traversal
import networkx as nx


def test_traversal_to_coalescent():
    edges = [(1,2), (1,3), (2,4), (2,5), (3,6), (3,7), (7,8), (9,10), (10,11), (11,12), (11,13)]
    times = {1:3, 2:1, 3:2, 4:0, 5:0, 6:0, 7:1, 8:0, 9:3, 10:2, 11:1, 12:0, 13:0}

    T = Traversal()
    T.generations = 3
    T.graph.add_edges_from(edges)
    nx.set_node_attributes(T.graph, times, 'time')

    C = T.to_coalescent()
    assert sorted(list(C.nodes)) == [1,2,3,4,5,6,8,11,12,13]


def test_traversal_to_coalescent_2():
    edges = [(1,2), (2,3), (3,4), (3,7), (7,8), (4,5), (5,6), (6,10), (6,9), (10,11)]
    times = {1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1, 8:0, 9:0, 10:1, 11:0}
    T = Traversal()
    T.generations = 7

    T.graph.add_edges_from(edges)
    nx.set_node_attributes(T.graph, times, 'time')

    C = T.to_coalescent()

    assert sorted(list(C.nodes)) == [3,6,8,9,11]
    assert C.get_edge_attr('weight') == {(3,6): 3, (3,8): 2, (6,9): 1, (6,11): 2}

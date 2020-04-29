import unittest
import numpy as np
import networkx as nx
from genealogy_aligner import Genealogy


class TestGenealogy(unittest.TestCase):

    def test_kinship_matrix(self):
        # Figure 5.1, Lange, 2ed (2003), p82
        G = Genealogy(3)
        G.add_couple(0, 1, 0)
        G.add_child(2, 0, 1, 1)
        G.add_child(3, 0, 1, 1)
        G.add_child(4, 2, 3, 2)
        G.add_child(5, 2, 3, 2)

        expected = np.array([[0.5  , 0.   , 0.25 , 0.25 , 0.25 , 0.25 ],
                             [0.   , 0.5  , 0.25 , 0.25 , 0.25 , 0.25 ],
                             [0.25 , 0.25 , 0.5  , 0.25 , 0.375, 0.375],
                             [0.25 , 0.25 , 0.25 , 0.5  , 0.375, 0.375],
                             [0.25 , 0.25 , 0.375, 0.375, 0.625, 0.375],
                             [0.25 , 0.25 , 0.375, 0.375, 0.375, 0.625]])
        assert np.allclose(G.kinship(), expected)


    def test_genealogy_simulation(self):
        G = Genealogy.from_founders(5, 5)
        assert(G.generations == 5)


    def test_genealogy_io(self):
        G = Genealogy.from_founders(5, 5)

        # roundtrip
        nx.write_adjlist(G, 'cached/simulation_1.txt', delimiter='\t')
        H = nx.read_adjlist('cached/simulation_1.txt', nodetype=int)

        assert(sorted(G.nodes) == sorted(H.nodes))

        
if __name__ == '__main__':
    unittest.main()

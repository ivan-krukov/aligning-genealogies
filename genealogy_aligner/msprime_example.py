import networkx as nx
import numpy as np
import msprime as msp
from genealogy_aligner import Genealogy

G = Genealogy.from_founders(10, 10, 2, 1)

with open('cached/1.txt', 'w') as out:
    print('ind\tmut\tfat', file=out)
    for n in G.nodes():
        p = list(G.predecessors(n))
        if p:
            print(n+1, p[0]+1, p[1]+1, sep='\t', file=out)
        else:
            print(n+1, 0, 0, sep='\t', file=out)

ped = msp.Pedigree.read_txt('1.txt')

sim = msp.simulate(len(G.probands()), model='wf_ped', pedigree=ped)

with open('fig/1.svg', 'w') as out:
	print(sim.first().draw(), file=out)

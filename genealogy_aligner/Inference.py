import csv
import networkx as nx
import tsinfer
import msprime as msp
from .Traversal import Traversal


def read_samples(filename):
    ''' Takes .csv file with genotype matrix and converts it to tsinfer SampleData format
        Args: filename (located in /data/samples)'''
    with open ( '../data/samples/' + filename + '.csv', encoding='utf-8', newline='') as file:
        data = list(csv.reader(file))
        length = len(data)
        sample_data = tsinfer.SampleData(sequence_length= length, num_flush_threads=2)
        for i in range(length):
            line =''.join(data[i])
            genotypes = []
            alleles = []
            al1 = line[0]
            for c in line:
                if c != al1:
                    genotypes.append ( 1 )
                    al2 = c
                else:
                    genotypes.append ( 0 )
            alleles.append(al1)
            alleles.append(al2)
            sample_data.add_site (i, genotypes, alleles )
    return sample_data


def infer_ts(filename):
    ''' Inferes tree sequence from genotype matrix
        Args: filename'''
    sample_data = read_samples(filename)
    inferred_ts = tsinfer.infer (sample_data)
    for tree in inferred_ts.trees ():
        print ( tree.draw ( format="unicode" ) )
    return inferred_ts


def convert_to_traversal(inferred_ts):
    ''' Converts the infered tree sequence to Traversal objects'''
    traversals = []
    for tree in inferred_ts.trees():
        t = Traversal()
        t.graph.ts_node_to_ped_node = {}
        for k,v in tree.parent_dict.items():
            t.graph.add_edge(k,v)
        if k in t.graph.nodes:
            t.graph.ts_node_to_ped_node = {k:v for k,v in tree.parent_dict.items()}

        nx.set_node_attributes ( t.graph,{n: inferred_ts.get_time ( n ) for n in t.nodes},'time' )
        traversals.append(t)
    return traversals


def infer_from_msprime(simulation):
    ''' Given msprime simulation results, obtains the corresponding inferred
        tree sequence using tsinfer
        Args: result - msprime output
    '''

    with tsinfer.SampleData (sequence_length=simulation.sequence_length, num_flush_threads=2) as sample_data:
        for var in simulation.variants ():
            sample_data.add_site ( var.site.position, var.genotypes, var.alleles )
    inferred_ts = tsinfer.infer (sample_data)
    return inferred_ts

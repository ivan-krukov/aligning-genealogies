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
    #for tree in inferred_ts.trees ():
        #print ( tree.draw ( format="unicode" ) )
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


def infer_from_msprime(ss=20, Ne=1e4, length=5e3, rho=2e-8,mu=2e-8, rs=10):
    ''' Given msprime simulation results, obtains the corresponding inferred
        tree sequence using tsinfer
        Args: sample_size (int): The number of sampled monoploid genomes.
              Ne (int): effective population size to use after the pedigree simulation is complete
              length (float): length of the genomic segment to simulate
              rho (float): recombination rate
              mu (float): mutation rate
              rs(int): random seed. If this is None, a random seed will be automatically generated.
    '''


    result = msp.simulate(sample_size = ss, Ne = Ne, length = length, recombination_rate=rho,
                          mutation_rate=mu, random_seed=rs)
    #print ( "Simulation done:", result.num_trees, "trees and", result.num_sites, "sites" )

    with tsinfer.SampleData (sequence_length=result.sequence_length, num_flush_threads=2) as sample_data:
        for var in result.variants ():
            sample_data.add_site ( var.site.position, var.genotypes, var.alleles )
    inferred_ts = tsinfer.infer (sample_data)
    return inferred_ts

#Running the script manually
'''if __name__ == "__main__":
    name = "sample1"
    inferred = infer_ts(name)
    print(convert_to_traversal(inferred))
    print(infer_from_msprime(20,1e4))
'''
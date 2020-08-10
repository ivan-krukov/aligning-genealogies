from collections import Counter
from genealogy_aligner import *
from genealogy_aligner.utils import invert_dictionary
import msprime as msp
import networkx as nx
import numpy.random as rnd


def ancestor_chain(T, node):
    chain = []
    p = T.first_parent_of(node)
    while p is not None:
        chain.append(p)
        node = p
        p = T.first_parent_of(node)
    return chain


class Agent:
    def __init__(self, ancestor_chain, ref_count=1):
        self.ancestor_chain = ancestor_chain
        self.ref_count = ref_count

    def __repr__(self):
        return f"Agent({self.ref_count}, {self.ancestor_chain})"


class AgentAligner:
    def __init__(self, P, progress=True):
        self.pedigree = P
        self.diploid_graph = DiploidGraph(P)
        self.depth = self.diploid_graph.get_node_attributes("time")
        K, idx = self.diploid_graph.kinship_lange(progress=progress)
        self.kinship = K
        self.kinship_idx = idx

    def sample_path(self, seed=None, Ne=1000, simplify=False):
        T = self.diploid_graph.sample_haploid_path(seed)
        if simplify:
            largest = max(nx.weakly_connected_components(T.graph), key=len)
            T.graph = nx.subgraph(T.graph, largest)
        C, msprime_idx = T.to_tree_sequence(simplify=True)
        msp_labels = invert_dictionary(msprime_idx, one_to_one=True)
        M = msp.simulate(from_ts=C, Ne=Ne, random_seed=seed, model="dtwf")
        O = Traversal.from_tree_sequence(M, msp_labels)
        return O, T

    def align(self, O, progress=True):
        probands = O.probands()
        Q = O.distance_matrix(kinship_like=True, progress=progress) / 2
        prob_idx = [self.kinship_idx[p] for p in probands]

        symmetries = Counter()

        agents = {}
        for p in probands:
            ancestors = ancestor_chain(O, p)
            agent = Agent(ancestors)
            agents[p] = agent

        parental_choices = {}
        node_alignments = {}
        symmetries = Counter()
        climber = Climber(self.diploid_graph, source=probands)
        for ped_node, ped_parents in climber:
            if not ped_parents:
                continue

            agent = agents[ped_node]
            if agent.ref_count > 1:
                ancestor = agent.ancestor_chain.pop(0)
                node_alignments[ancestor] = ped_node

            gen_parent = agent.ancestor_chain[0]

            d = self.depth[ped_node] + 1

            left_stat = self.kinship[self.kinship_idx[ped_parents[0]], prob_idx]
            right_stat = self.kinship[self.kinship_idx[ped_parents[1]], prob_idx]
            up_stat = Q[gen_parent, probands].todense()

            # calc with dot-products
            left = up_stat @ left_stat
            right = up_stat @ right_stat

            if left > right:
                choice = ped_parents[0]
            elif left < right:
                choice = ped_parents[1]
            else:
                rch = rnd.choice(2)
                choice = ped_parents[rch]
                symmetries[d] += 1
            parental_choices[ped_node] = choice

            climber.queue(choice)

            if choice in agents:
                agents[choice].ref_count += 1
            else:
                a = Agent(agent.ancestor_chain)
                agents[choice] = a

        return parental_choices, node_alignments, symmetries

    def evaluate(self, choices, T):
        correct_ind = Counter()
        correct_ploid = Counter()
        total = Counter()
        for node, inferred_parent in choices.items():
            d = self.depth[node] + 1
            try:
                true_parent = T.first_parent_of(node)
                inferred_ind = (inferred_parent + 1) // 2
                true_ind = (true_parent + 1) // 2
                if inferred_ind == true_ind:
                    correct_ind[d] += 1
                if inferred_parent == true_parent:
                    correct_ploid[d] += 1
            except:
                pass
            finally:
                total[d] += 1
        return correct_ploid, correct_ind, total

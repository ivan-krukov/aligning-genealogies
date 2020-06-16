from gensim.models.poincare import PoincareModel
import networkx as nx
import copy
import numpy as np

from .evaluation import accuracy
from .utils import greedy_matching
from .Drawing import draw_aligner


class Aligner(object):

    def __init__(self, ped, ts):
        """

        :param ped: A pedigree
        :param ts: A traversal object
        """

        self.ped = ped
        self.ts = ts

        self.ped_probands = self.ped.probands()
        self.ped_nonprobands = list(set(self.ped.nodes) - set(self.ped_probands))
        self.ts_probands = self.ts.probands()
        self.ts_nonprobands = list(set(self.ts.nodes) - set(self.ts_probands))

        self.true_ts_node_to_ped_node = copy.copy(ts.ts_node_to_ped_node)

        for n_ts in (set(self.ts.nodes) - set(ts.ts_node_to_ped_node)):
            self.true_ts_node_to_ped_node[n_ts] = None

        self.true_ped_node_to_ts_edge = copy.copy(ts.ped_node_to_ts_edge)

        self.ts_proband_to_ped_proband = {
            n_ts: n_ped
            for n_ts, n_ped in self.true_ts_node_to_ped_node.items()
            if n_ts in self.ts_probands
        }

        self.pred_ts_node_to_ped_node = None
        self.pred_ped_node_to_ts_edge = None

    def align(self):
        raise NotImplementedError

    def evaluate(self):

        if self.pred_ts_node_to_ped_node is None:
            raise Exception("No pairs are predicted to be aligned. Call `align` first.")

        metrics = {
            'accuracy': accuracy(self.true_ts_node_to_ped_node.items(),
                                 self.pred_ts_node_to_ped_node.items())
        }

        return metrics

    def draw(self, use_predicted=False, exclude_probands=True,
             use_ped_labels=True, ped_kwargs=None, ts_kwargs=None, **kwargs):

        if use_predicted and (self.pred_ts_node_to_ped_node is None):
            raise Exception("You must call .align() in order to view predicted alignments.")

        if use_predicted:
            anchor_links = self.pred_ts_node_to_ped_node
        else:
            anchor_links = self.true_ts_node_to_ped_node

        if exclude_probands:
            anchor_links = {k: v for k, v in anchor_links.items()
                            if v not in self.ped_probands}

        anchor_links = {k: v for k, v in anchor_links.items()
                        if k is not None and v is not None}

        if use_ped_labels:
            if ts_kwargs is None:
                ts_kwargs = {'label_dict': self.ts.ts_node_to_ped_node}
            else:
                ts_kwargs['label_dict'] = self.ts.ts_node_to_ped_node

        draw_aligner(self.ped, self.ts, anchor_links,
                     ped_kwargs=ped_kwargs, ts_kwargs=ts_kwargs, **kwargs)


class MatchingAligner(Aligner):
    """
    This is the parent class for all matching Aligners.
    It implements the `match` method, which serves as the main
    step in the discrete alignment process.
    """

    def __init__(self, ped, ts):
        super().__init__(ped, ts)

    @staticmethod
    def match(ped_dict, ts_dict):
        """
        The match method takes 2 dictionaries from the
        pedigree and tree sequence graphs.
        Each dictionary consists of the following key and value pairs
        * key: The key is the name of the non-aligned node
        * value: A set of the closest aligned nodes to it. What is
        meant by 'closest' and how many items in the set are left
        for the child classes to decide.

        For the matching to work, ensure that the set of closest
        haplotypes in the tree sequence dictionary are mapped
        to the corresponding subjects
        using the self.ts_proband_to_ped_proband dictionary.

        :param ped_dict: Pedigree dictionary
        :param ts_dict: Tree sequence dictionary
        :return:
        """

        pairs = []
        scores = []

        for n_ped, n_ped_el in ped_dict.items():
            for n_ts, n_ts_el in ts_dict.items():
                score = (len(n_ped_el.intersection(n_ts_el)) /
                         len(n_ped_el.union(n_ts_el)))

                pairs += [[n_ts, n_ped]]
                scores.append(score)

        return dict(
            greedy_matching(np.array(pairs).T, np.array(scores))
        )

    def harmonize(self, g_matching):
        """
        Helper method to harmonize greedy alignments.
        :param g_matching:
        :return:
        """

        # -------------------------------------------------------
        # ** Step 1 **: Dealing with out-of-pedigree nodes
        # = = = = = = = = =
        # Plan of action: Check Two cases:
        # [1] if a node is matched to a founder node
        # in the pedigree, then all of its ancestors should be
        # out-of-pedigree nodes.
        # [2] if 2 nodes are connected in the tree sequence,
        # they must be connected in the pedigree. Otherwise,
        # their ancestors in the tree sequence are out-of-pedigree
        # nodes.

        # Case [1]:
        ped_founders = self.ped.founders()

        for ts_n, ped_n in g_matching.items():
            if ped_n in ped_founders:
                ts_n_pred = self.ts.predecessors(ts_n)

                while len(ts_n_pred) > 0:
                    curr_node = ts_n_pred[0]
                    g_matching[curr_node] = None
                    ts_n_pred.extend(self.ts.predecessors(curr_node))
                    ts_n_pred = ts_n_pred[1:]

        # Case [2]:
        for ts_n1, ped_n1 in g_matching.items():

            if ped_n1 is None:
                continue

            for ts_n2 in self.ts.siblings(ts_n1):
                if ts_n2 in g_matching and g_matching[ts_n2] is not None:
                    ped_n2 = g_matching[ts_n2]
                    if nx.lowest_common_ancestor(self.ped.graph, ped_n1, ped_n2) is None:

                        for ts_n in [ts_n1, ts_n2]:
                            ts_n_pred = self.ts.predecessors(ts_n)
                            while len(ts_n_pred) > 0:
                                curr_node = ts_n_pred[0]
                                g_matching[curr_node] = None
                                ts_n_pred.extend(self.ts.predecessors(curr_node))
                                ts_n_pred = ts_n_pred[1:]

        # -------------------------------------------------------
        # ** Step 2 **: Checking time-order of mappings
        # = = = = = = = = =
        # Plan of action:

        # -------------------------------------------------------
        # ** Step 3 **: Number of mappings checks
        # = = = = = = = = =
        # Plan of action:

        return g_matching


class DescMatchingAligner(MatchingAligner):

    def __init__(self, ped, ts, iterative=True, climb_up_step=0):
        super().__init__(ped, ts)
        self.iterative = iterative
        self.climb_up_step = climb_up_step

    def get_ntps(self):

        ped_ntp = self.ped.get_probands_under(climb_up_step=self.climb_up_step)
        ts_ntp = self.ts.get_probands_under(climb_up_step=self.climb_up_step)

        # For the intersection/union metric to work, we need to update
        # the ts_ntp data structure with the pedigree node IDs instead
        # of the haplotype IDs:

        for n, n_set in ts_ntp.items():
            ts_ntp[n] = set([self.ts_proband_to_ped_proband[pid]
                             for pid in n_set])

        return ped_ntp, ts_ntp

    def align(self):

        ped_ntp, ts_ntp = self.get_ntps()

        mapped_ts = {tsn: self.true_ts_node_to_ped_node[tsn]
                     for tsn in self.ts_probands}
        unmapped_ts = self.ts_nonprobands

        ts_node_time = self.ts.get_node_attributes('time')

        while len(unmapped_ts) > 0:

            if self.iterative:
                # Find the closest nodes to the mapped nodes by time difference:
                unmapped_time = {n: ts_node_time[n] for n in unmapped_ts}
                min_t = min(unmapped_time.values())
                closest_unmapped = [n for n in unmapped_time if ts_node_time[n] == min_t]
            else:
                closest_unmapped = unmapped_ts

            mapped_ts.update(
                self.match(
                    {n_ped: ped_ntp[n_ped] for n_ped in self.ped_nonprobands},
                    {n_ts: ts_ntp[n_ts] for n_ts in closest_unmapped}
                )
            )

            unmapped_ts = [n for n in self.ts.nodes if n not in mapped_ts.keys()]

        self.pred_ts_node_to_ped_node = self.harmonize(mapped_ts)

        return self.pred_ts_node_to_ped_node


class PoincareAligner(MatchingAligner):

    def __init__(self, ped, ts, iterative=True, size=10, k=5):
        super().__init__(ped, ts)
        self.iterative = True
        self.size = size
        self.k = k

    def align(self):

        ped_model = PoincareModel(self.ped.edges, size=self.size, negative=2)
        ts_model = PoincareModel(self.ts.edges, size=self.size, negative=2)

        mapped_ts = {tsn: self.true_ts_node_to_ped_node[tsn]
                     for tsn in self.ts_probands}
        unmapped_ts = self.ts_nonprobands

        ts_node_time = self.ts.get_node_attributes('time')

        while len(unmapped_ts) > 0:

            if self.iterative:
                # Find the closest nodes to the mapped nodes by time difference:
                unmapped_time = {n: ts_node_time[n] for n in unmapped_ts}
                min_t = min(unmapped_time.values())
                closest_unmapped = [n for n in unmapped_time if ts_node_time[n] == min_t]
            else:
                closest_unmapped = unmapped_ts

            ped_sim = {n: set(sorted(self.ped_probands,
                                     key=lambda k: ped_model.kv.similarity(k, n),
                                     reverse=True)[:self.k])
                       for n in self.ped_nonprobands}
            ts_sim = {n: sorted(self.ts_probands,
                                key=lambda k: ts_model.kv.similarity(k, n),
                                reverse=True)[:self.k]
                      for n in closest_unmapped}

            ts_sim = {k: set([self.ts_proband_to_ped_proband[i] for i in v])
                      for k, v in ts_sim.items()}

            mapped_ts.update(
                self.match(
                    ped_sim, ts_sim
                )
            )

            unmapped_ts = [n for n in self.ts.nodes if n not in mapped_ts.keys()]

        self.pred_ts_node_to_ped_node = self.harmonize(mapped_ts)

        return self.pred_ts_node_to_ped_node

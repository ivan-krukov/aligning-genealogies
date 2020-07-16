from __future__ import absolute_import
import networkx as nx
import copy
import numpy as np

from .evaluation import accuracy, simple_symmetries
from .utils import greedy_matching, soft_ordering, invert_dictionary
from .Drawing import draw_aligner


class Aligner(object):

    def __init__(self, ped, ts):
        """

        :param ped: A pedigree
        :param ts: A traversal object
        """

        self.ped = ped
        self.ts = ts

        if self.ts.haploid_probands:
            self.ploidy = 1
        else:
            self.ploidy = 2

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

        metrics = {}

        if self.pred_ts_node_to_ped_node is None:
            raise Exception("No pairs are predicted to be aligned. Call `align` first.")
        else:
            metrics['Node-Node Matching Accuracy'] = accuracy(
                self.true_ts_node_to_ped_node.items(),
                self.pred_ts_node_to_ped_node.items()
            )

        metrics['Proportion of Simple Symmetries'] = simple_symmetries(
            self.ped,
            self.true_ts_node_to_ped_node,
            self.pred_ts_node_to_ped_node
        )

        if self.pred_ped_node_to_ts_edge is None:
            if self.true_ped_node_to_ts_edge is None:
                metrics['Node-Edge Matching Accuracy'] = None
            elif len(self.true_ped_node_to_ts_edge) == 0:
                metrics['Node-Edge Matching Accuracy'] = None
            else:
                metrics['Node-Edge Matching Accuracy'] = 0.0
        else:
            metrics['Node-Edge Matching Accuracy'] = accuracy(
                self.true_ped_node_to_ts_edge.items(),
                self.pred_ped_node_to_ts_edge.items()
            )

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

    def match(self, ped_dict, ts_dict):
        """
        Applies greedy matching to the scoring results
        """

        return dict(
            greedy_matching(*self.score(ped_dict, ts_dict))
        )

    def soft_order(self, ped_dict, ts_dict):
        """
        Applies soft ordering to the scoring results
        """

        return soft_ordering(
            *self.score(ped_dict, ts_dict), return_confidence=True
        )

    @staticmethod
    def score(ped_dict, ts_dict):
        """
        The score method takes 2 dictionaries from the
        pedigree and tree sequence graphs.
        Each dictionary consists of the following key and value pairs
        * key: The key is the name of the non-aligned node
        * value: A set of the closest aligned nodes to it. What is
        meant by 'closest' and how many items in the set are left
        for the child classes to define.

        For the scoring to work, ensure that the set of closest
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

        return np.array(pairs).T, np.array(scores)

    def complete_paths(self):

        m_dict = self.pred_ts_node_to_ped_node.copy()
        m_dict.update(self.ts_proband_to_ped_proband)

        rev_m_dict = invert_dictionary(m_dict)

        matched_nodes = rev_m_dict.keys()

        if len(matched_nodes) == 0:
            raise Exception("No matched nodes. Call `.align()` first.")

        nn_nodes_to_edges = {}

        for n in self.ped.nodes:

            if n in rev_m_dict:
                if len(rev_m_dict[n]) >= self.ploidy:
                    continue

            # [1] Simple case: If a pedigree node is sandwiched between 2
            # matched nodes, and there's an edge between those 2 nodes in the
            # tree sequence, then assign the pedigree node to that edge.

            pred = self.ped.predecessors(n)
            succ = self.ped.successors(n)

            for p in pred:
                for s in succ:
                    if p in matched_nodes and s in matched_nodes:
                        for ts_n1 in rev_m_dict[p]:
                            for ts_n2 in rev_m_dict[s]:
                                if ts_n2 in self.ts.successors(ts_n1):
                                    nn_nodes_to_edges[n] = (ts_n1, ts_n2)

        self.pred_ped_node_to_ts_edge = nn_nodes_to_edges

        return self.pred_ped_node_to_ts_edge

    def harmonize(self, g_matching, s_matching):
        """
        Helper method to harmonize greedy alignments.
        """

        # -------------------------------------------------------
        # ** Step 0 **: Number of mappings checks
        # = = = = = = = = =
        # Plan of action: In the diploid case, each individual in the pedigree is assigned to only 2
        # different nodes in the tree sequence. In the haploid case, it is 1.

        # invert the matching dictionary:
        rev_g_matching = invert_dictionary(g_matching)

        for ped_n, ts_n_li in rev_g_matching.items():
            if ped_n is not None and len(ts_n_li) > self.ploidy:

                ts_n_li = sorted(
                    ts_n_li,
                    key=lambda x: [sc for p_n, sc in s_matching[x] if p_n == ped_n][0],
                    reverse=True
                )

                for ts_n in ts_n_li[2:]:
                    for ped_n2, sc in s_matching[ts_n]:
                        if ped_n2 in rev_g_matching:
                            if ped_n2 != ped_n and len(rev_g_matching[ped_n2]) < self.ploidy:
                                g_matching[ts_n] = ped_n2

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

        for ts_n in list(g_matching):
            ped_n = g_matching[ts_n]
            if ped_n in ped_founders:
                ts_n_pred = self.ts.predecessors(ts_n)

                while len(ts_n_pred) > 0:
                    curr_node = ts_n_pred[0]
                    g_matching[curr_node] = None
                    ts_n_pred.extend(self.ts.predecessors(curr_node))
                    ts_n_pred = ts_n_pred[1:]

        # Case [2]:
        for ts_n1 in list(g_matching):

            ped_n1 = g_matching[ts_n1]

            if ped_n1 is None:
                continue

            for ts_n2 in self.ts.siblings(ts_n1):
                if ts_n2 in g_matching and g_matching[ts_n2] is not None:
                    ped_n2 = g_matching[ts_n2]
                    '''
                    for pn1 in list(ped_n1):
                        for pn2 in list(ped_n2):
                            if nx.lowest_common_ancestor ( self.ped.graph, pn1, pn2 ) is None:
                    '''
                    if nx.lowest_common_ancestor(self.ped.graph, ped_n1, ped_n2) is None:

                        for ts_n in [ts_n1, ts_n2]:
                            ts_n_pred = self.ts.predecessors(ts_n)
                            while len(ts_n_pred) > 0:
                                curr_node = ts_n_pred[0]
                                g_matching[curr_node] = None
                                ts_n_pred.extend(self.ts.predecessors(curr_node))
                                ts_n_pred = ts_n_pred[1:]

        # -------------------------------------------------------
        # ** Step 2 **: Logical check for diploid matchings:
        # = = = = = = = = =
        # Plan of action: In the diploid case, if an individual in the pedigree is assigned to 2
        # nodes in the tree sequence, then those nodes cannot be ancestral to each
        # other (they have to be on separate branches of the tree).
        #
        # TODO: This code is broken and causes infinite loops.

        """
        if self.ploidy == 2:

            # Invert the matching dictionary
            rev_g_matching = invert_dictionary(g_matching)

            for ped_n, ts_n_li in rev_g_matching.items():
                # If there are 2 items in the list:
                if len(ts_n_li) == 2:
                    # If the first item is ancestral to the second:
                    if ts_n_li[0] in self.ts.predecessors(ts_n_li[1],
                                                          k=None,
                                                          include_intermediates=True):
                        del g_matching[ts_n_li[0]]
                    # If the second item is ancestral to the first:
                    elif ts_n_li[1] in self.ts.predecessors(ts_n_li[0],
                                                            k=None,
                                                            include_intermediates=True):
                        del g_matching[ts_n_li[1]]
        """
        # -------------------------------------------------------
        # ** Step 3 **: Checking time-order of mappings
        # = = = = = = = = =
        # Plan of action: For an aligned pair (n_ts, n_ped),
        # make sure that their successors and predecessors preserve their time-ordering.

        '''
        for ts_n, ped_n in g_matching.items ():

            ts_n_time = self.ts.get_node_attributes ( 'time', ts_n )
            ped_n_time = self.ped.get_node_attributes ( 'time', ped_n )

            ts_pred_list = self.ts.predecessors ( ts_n )
            ts_succ_list = self.ts.successors ( ts_n )

            # Preserving time ordering of predecessors: Regardless of how we infer time,
            # times of predecessor at pedigree and tree sequence, must be both larger or
            # both smaller than current time.

            for ts_n_pred in ts_pred_list:
                for ts_n_succ in ts_succ_list:
                    if (ts_n_pred is None) or (ts_n_succ is None):
                        continue
                    else:
                        if ts_n_pred in g_matching and g_matching[ts_n_pred] is not None:
                            ped_n_pred = list ( g_matching[ts_n_pred] )

                        if ts_n_succ in g_matching and g_matching[ts_n_succ] is not None:
                            ped_n_succ = list ( g_matching[ts_n_succ] )

                        ts_n_pred_time = self.ts.get_node_attributes ( 'time', ts_n_pred )
                        ped_n_pred_time = []
                        for i in range ( len ( ped_n_pred ) ):
                            ped_n_pred_time[i] = self.ts.get_node_attributes ( 'time', ped_n_pred[i] )

                        if ((ts_n_pred_time - ts_n_time) *
                                (ped_n_pred_time[i] - ped_n_time) < 0):
                            del ped_n_pred[i]

                        ts_n_succ_time = self.ts.get_node_attributes ( 'time', ts_n_succ )
                        ped_n_succ_time = []
                        for i in range ( len ( ped_n_succ ) ):
                            ped_n_succ_time[i] = self.ts.get_node_attributes ( 'time', ped_n_succ[i] )

                        if ((ts_n_succ_time - ts_n_time) *
                                (ped_n_succ_time[i] - ped_n_time) < 0):
                            del ped_n_pred[i]

                        if not (((ts_n_succ_time > ts_n_time) and (ts_n_time > ts_n_pred_time))
                                or ((ts_n_succ_time < ts_n_time) and (ts_n_time < ts_n_pred_time))):
                            del ped_n_succ, ped_n_pred
        '''
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

        scoring_dict = self.soft_order(
            {n_ped: ped_ntp[n_ped] for n_ped in self.ped_nonprobands},
            {n_ts: ts_ntp[n_ts] for n_ts in self.ts_nonprobands}
        )

        ts_node_time = self.ts.get_node_attributes('time')

        while len(unmapped_ts) > 0:

            if self.iterative:
                # Find the closest nodes to the mapped nodes by time difference:
                unmapped_time = {n: ts_node_time[n] for n in unmapped_ts}
                min_t = min(unmapped_time.values())
                closest_unmapped = [n for n in unmapped_time if ts_node_time[n] == min_t]
            else:
                closest_unmapped = unmapped_ts

            dicts = (
                {n_ped: ped_ntp[n_ped] for n_ped in self.ped_nonprobands},
                {n_ts: ts_ntp[n_ts] for n_ts in closest_unmapped}
            )

            mapped_ts.update(self.match(*dicts))
            mapped_ts = self.harmonize(mapped_ts, scoring_dict)

            unmapped_ts = [n for n in self.ts.nodes if n not in mapped_ts.keys()]

        self.pred_ts_node_to_ped_node = mapped_ts

        return self.pred_ts_node_to_ped_node

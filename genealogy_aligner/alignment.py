from evaluation import accuracy
from utils import greedy_matching
import numpy as np


class Aligner(object):

    def __init__(self, ped, ts, aligned_pairs=None):
        """

        :param ped: A pedigree
        :param ts: A tree sequence
        :param aligned_pairs: A list of tuples of the form (n_ts, n_ped) where `n_ts` is the
        node in the tree sequence and `n_ped` is the node in the pedigree.
        """

        self.ped = ped
        self.ts = ts
        if aligned_pairs is None:
            self.aligned_pairs = [(n_ts, n_ped)
                                  for n_ped in ped.nodes
                                  for n_ts in ts.nodes
                                  if n_ped == n_ts]
        else:
            self.aligned_pairs = aligned_pairs

        self.pred_pairs = None

    def align(self):
        raise NotImplementedError

    def evaluate(self):

        if self.pred_pairs is None:
            raise Exception("No pairs are predicted to be aligned. Call `align` first.")

        metrics = {
            'accuracy': accuracy(self.aligned_pairs, self.pred_pairs)
        }

        return metrics


class DescMatchingAligner(Aligner):
    """
    Descendant Matching Aligner: Aligns nodes based on similarity of their
    sets of descendants (probands).
    """

    def __init__(self, ped, ts, aligned_pairs=None, climb_up_step=0):
        super().__init__(ped, ts, aligned_pairs)
        self.climb_up_step = climb_up_step

    def align(self):

        ped_ntp = self.ped.get_probands_under(climb_up_step=self.climb_up_step)
        ts_ntp = self.ts.get_probands_under(climb_up_step=self.climb_up_step)

        pairs = []
        scores = []

        ped_prob = self.ped.probands()
        ts_prob = self.ts.probands()

        for i, n_ped in enumerate(self.ped.nodes):
            for j, n_ts in enumerate(self.ts.nodes):

                if n_ped in ped_prob or n_ts in ts_prob:
                    continue

                score = (len(ped_ntp[n_ped].intersection(ts_ntp[n_ts])) /
                         len(ped_ntp[n_ped].union(ts_ntp[n_ts])))

                pairs += [[n_ts, n_ped]]
                scores.append(score)

        self.pred_pairs = greedy_matching(np.array(pairs).T, np.array(scores))

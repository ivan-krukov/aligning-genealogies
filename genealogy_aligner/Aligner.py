import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .evaluation import accuracy
from .utils import greedy_matching


class Aligner(object):

    def __init__(self, ped, ts):
        """

        :param ped: A pedigree
        :param ts: A traversal object
        :param aligned_pairs: A list of tuples of the form (n_ts, n_ped) where `n_ts` is the
        node in the tree sequence and `n_ped` is the node in the pedigree.
        """

        self.ped = ped
        self.ts = ts

        self.true_ts_node_to_ped_node = ts.ts_node_to_ped_node.items()
        self.true_ped_node_to_ts_edge = ts.ped_node_to_ts_edge

        self.pred_ts_node_to_ped_node = None
        self.pred_ped_node_to_ts_edge = None

    def align(self):
        raise NotImplementedError

    def evaluate(self):

        if self.pred_ts_node_to_ped_node is None:
            raise Exception("No pairs are predicted to be aligned. Call `align` first.")

        metrics = {
            'accuracy': accuracy(self.true_ts_node_to_ped_node,
                                 self.pred_ts_node_to_ped_node)
        }

        return metrics

    def draw(self, use_predicted=False,
             figsize=(16, 8), labels=True,
             ped_node_color=None, ts_node_color=None,
             use_ped_labels=True):

        if use_predicted and (self.pred_ts_node_to_ped_node is None):
            raise Exception("You must call .align() in order to view predicted alignments.")

        fig, axes = plt.subplots(ncols=2, figsize=figsize)

        self.ped.draw(ax=axes[0], node_color=ped_node_color, labels=labels)

        if use_ped_labels:
            self.ts.draw(ax=axes[1], node_color=ts_node_color, labels=labels,
                         label_dict=self.ts.ts_node_to_ped_node)
        else:
            self.ts.draw(ax=axes[1], node_color=ts_node_color, labels=labels)

        ped_layout = self.ped.get_graphviz_layout()
        ts_layout = self.ts.get_graphviz_layout()

        if use_predicted:
            align_map = self.pred_ts_node_to_ped_node
        else:
            align_map = self.true_ts_node_to_ped_node

        # Transform figure:
        ax0tr = axes[0].transData
        ax1tr = axes[1].transData
        figtr = fig.transFigure.inverted()

        for ts_n, ped_n in align_map:
            # 2. Transform arrow start point from axis 0 to figure coordinates
            ptB = figtr.transform(ax0tr.transform(ped_layout[ped_n]))
            # 3. Transform arrow end point from axis 1 to figure coordinates
            ptE = figtr.transform(ax1tr.transform(ts_layout[ts_n]))
            # 4. Create the patch
            arrow = matplotlib.patches.FancyArrowPatch(
                ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
                fc="grey", connectionstyle="arc3,rad=0.3", arrowstyle='-', alpha=0.3,
                shrinkA=10, shrinkB=10, linestyle='dashed'
            )
            # 5. Add patch to list of objects to draw onto the figure
            fig.patches.append(arrow)


class DescMatchingAligner(Aligner):
    """
    Descendant Matching Aligner: Aligns nodes based on similarity of their
    sets of descendants (probands).
    """

    def __init__(self, ped, ts, climb_up_step=0):
        super().__init__(ped, ts)
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

        self.pred_ts_node_to_ped_node = greedy_matching(np.array(pairs).T, np.array(scores))

        return self.pred_ts_node_to_ped_node

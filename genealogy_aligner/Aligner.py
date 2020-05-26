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
        """

        self.ped = ped
        self.ts = ts

        self.true_ts_node_to_ped_node = ts.ts_node_to_ped_node
        self.true_ped_node_to_ts_edge = ts.ped_node_to_ts_edge

        self.ts_proband_to_ped_proband = {
            n_ts: n_ped
            for n_ts, n_ped in self.true_ts_node_to_ped_node.items()
            if n_ts in self.ts.probands()
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

    def draw(self, use_predicted=False,
             figsize=(16, 8), labels=True,
             ped_node_color=None, ts_node_color=None,
             use_ped_labels=True,
             edge_color_map='nipy_spectral'):

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

        cmap = matplotlib.cm.get_cmap(edge_color_map)

        align_map = [(ts_n, ped_n) for ts_n, ped_n in align_map.items()
                     if ped_n not in self.ped.probands()]

        for i, (ts_n, ped_n) in enumerate(align_map):
            # 2. Transform arrow start point from axis 0 to figure coordinates
            ptB = figtr.transform(ax0tr.transform(ped_layout[ped_n]))
            # 3. Transform arrow end point from axis 1 to figure coordinates
            ptE = figtr.transform(ax1tr.transform(ts_layout[ts_n]))
            # 4. Create the patch
            arrow = matplotlib.patches.FancyArrowPatch(
                ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
                color=cmap(i/len(align_map)), connectionstyle="arc3,rad=0.3",
                arrowstyle='-', alpha=0.3,
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

        self.pred_ts_node_to_ped_node = dict(
            greedy_matching(np.array(pairs).T, np.array(scores))
        )

        return self.pred_ts_node_to_ped_node


class IterativeMatchingAligner(DescMatchingAligner):

    def __init__(self, ped, ts):
        super().__init__(ped, ts)

    def align(self):

        ped_ntp, ts_ntp = self.get_ntps()
        ped_non_proband_nodes = [n for n in self.ped.nodes
                                 if n not in self.ped.probands()]

        mapped_ts = {tsn: self.true_ts_node_to_ped_node[tsn]
                     for tsn in self.ts.probands()}
        unmapped_ts = [n for n in self.ts.nodes if n not in mapped_ts.keys()]

        ts_node_time = self.ts.get_node_attributes('time')

        while len(unmapped_ts) > 0:

            unmapped_time = {n: ts_node_time[n] for n in unmapped_ts}

            # Find the closest nodes to the mapped nodes by time difference:
            min_t = min(unmapped_time.values())
            closest_unmapped = [n for n in unmapped_time if ts_node_time[n] == min_t]

            pairs = []
            scores = []

            for i, n_ped in enumerate(ped_non_proband_nodes):
                for j, n_ts in enumerate(closest_unmapped):

                    score = (len(ped_ntp[n_ped].intersection(ts_ntp[n_ts])) /
                             len(ped_ntp[n_ped].union(ts_ntp[n_ts])))

                    pairs += [[n_ts, n_ped]]
                    scores.append(score)

            mapped_ts.update(dict(
                greedy_matching(np.array(pairs).T, np.array(scores))
            ))

            unmapped_ts = [n for n in self.ts.nodes if n not in mapped_ts.keys()]

        self.pred_ts_node_to_ped_node = mapped_ts

        return self.pred_ts_node_to_ped_node

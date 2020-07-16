import numpy as np
from .utils import soft_ordering


def accuracy(pos_anchors, pred_anchors):
    """
    Defined as in Equation (13) of Trung et al.
    https://www.sciencedirect.com/science/article/pii/S0957417419305937

    :param pos_edge_index: list of tuples of matching nodes in graphs 1 and 2
    :param pred_edge_index: list of tuples of predicted matching nodes in graphs 1 and 2

    """
    return float(len(set(pos_anchors).intersection(set(pred_anchors)))) / len(pos_anchors)


def MAP(pairs, sim_score):
    s_order = soft_ordering(pairs, sim_score)
    return np.mean([1. / (li.index(n) + 1) for n, li in s_order.items() if n in li])


def simple_symmetries(ped, true_ts_to_ped, pred_ts_to_ped):
    """
    This function takes a pedigree object, a true set of alignments
    and a predicted set of alignments. If there are mistakes in the
    predicted set of alignments, it counts the proportion of those
    mistakes due to simple symmetries (here defined as mixing the
    identities of the 2 spouses).
    """

    count_mistakes = 0
    count_symmetries = 0

    for ts_n, pred_ped_n in pred_ts_to_ped.items():
        true_ped_n = true_ts_to_ped[ts_n]
        if pred_ped_n != true_ped_n:
            count_mistakes += 1

            if pred_ped_n is not None and true_ped_n is not None:
                if true_ped_n in ped.pairs(pred_ped_n):
                    count_symmetries += 1

    return float(count_symmetries) / count_mistakes



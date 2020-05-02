from utils import greedy_matching, soft_ordering
import numpy as np


def accuracy(pairs, sim_score):
    greedy_res = greedy_matching(pairs, sim_score)
    return sum([c == g for c, g in greedy_res]) / len(greedy_res)


def MAP(pairs, sim_score):
    s_order = soft_ordering(pairs, sim_score)
    return np.mean([1. / (li.index(n) + 1) for n, li in s_order.items() if n in li])


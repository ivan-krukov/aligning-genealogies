# re-implementation of the GENLIB kinship calculation

import numpy as np
from tqdm import tqdm
from genealogy_aligner import Pedigree

def get_index(table, query):
    # return index of each query in table
    # assumes table is sorted
    # returns -1 for query of 0
    idx = np.searchsorted(table, query)
    idx[query == 0] = -1
    return idx


def kinship_matrix(ind_id, mat_id, pat_id, depth):
    n = len(ind_id)
    assert n == len(mat_id)
    assert n == len(pat_id)

    K = np.eye(n+1) * 0.5
    K[-1,-1] = 0 # last row and column are for founders
    if n == 1:
        return K

    m = get_index(ind_id, mat_id)
    p = get_index(ind_id, pat_id)

    # flip!
    depth = np.abs(depth - max(depth))
    for d in tqdm(range(1, max(depth)+1)):
        idx = np.flatnonzero(depth == d) 
        K[idx, :] = (K[m[idx], :] + K[p[idx], :]) / 2
        K[:, idx] = (K[:, m[idx]] + K[:, p[idx]]) / 2
        K[idx,idx] = (1 + K[m[idx], p[idx]]) / 2

    return K[:-1,:-1]

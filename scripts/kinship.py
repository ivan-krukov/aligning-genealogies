# re-implementation of the GENLIB kinship calculation

import numpy as np
from tqdm import tqdm


def kindepth(ind_id, mat_id, pat_id):
    n = len(ind_id)
    assert n == len(mat_id)
    assert n == len(pat_id)

    depth = np.zeros(n, dtype=int)
    if n == 1:
        return depth

    parents = ind_id[(mat_id == 0) & (pat_id == 0)]

    for i in np.arange(1, n+1):
        # all the individuals whose parents are founders
        children = np.isin(mat_id, parents) | np.isin(pat_id, parents)
        if i == n:
            raise RuntimeError("Impossible pedigree - someone is their own ancestor")
        if np.any(children):
            depth[children] = i
            parents = ind_id[children]
        else: 
            break

    return depth

def get_index(table, query):
    # return index of each query in table
    # assumes table is sorted
    # returns -1 for query of 0
    idx = np.searchsorted(table, query)
    idx[query == 0] = -1
    return idx


def kinmat(ind_id, mat_id, pat_id):
    n = len(ind_id)
    assert n == len(mat_id)
    assert n == len(pat_id)

    K = np.eye(n+1) * 0.5
    K[-1,-1] = 0 # last row and column are for founders
    if n == 1:
        return K

    depth = kindepth(ind_id, mat_id, pat_id)
    m = get_index(ind_id, mat_id)
    p = get_index(ind_id, pat_id)

    print("iterating")
    for d in tqdm(range(1, max(depth)+1)):
        idx = np.flatnonzero(depth == d) 
        K[idx, :] = (K[m[idx], :] + K[p[idx], :]) / 2
        K[:, idx] = (K[:, m[idx]] + K[:, p[idx]]) / 2
        for j in tqdm(idx):
            K[j,j] = (1 + K[m[j], p[j]]) / 2

    return K[:-1,:-1]

def kindict(df):
    """Compute the kinship matrix for the genealogy"""
    n = len(df)
    K = dict()
    midx = get_index(df.ind, df.mother)
    pidx = get_index(df.ind, df.father)
    
    for i, (_, ind) in enumerate(tqdm(df.iterrows(), total=n)):
        if (ind.mother == 0) and (ind.father == 0):
            K[(i,i)] = 0.5
        else:
            K[(i,i)] = 0.5 + (K[(midx[i],pidx[i])]/2)
        
        for j in range(i+1, n):
            if (ind.mother != 0) and (ind.father != 0):
                K[(i,j)] = (K[(i,midx[j])]/2) + (K[(i,pidx[j])]/2)
    return K
        

pedigree = dict(
        ind_id = np.arange(1, 15) * 10,
        pat_id = np.array([4,5,8,9,11,13,0,0,0,0,8,0,0,0]) * 10,
        mat_id = np.array([3,6,7,10,12,14,0,0,0,0,7,0,0,0]) * 10)


kd = kindepth(**pedigree)
assert np.allclose(kd, [2, 3, 1, 1, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0])

K = kinmat(**pedigree)


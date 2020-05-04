import numpy as np


def create_similarity_matrix(gg, ct, climb_up_step=0):

    gg_nodes = list(gg.nodes())
    ct_nodes = list(ct.nodes())

    gg_ntp = gg.get_probands_under(climb_up_step=climb_up_step)
    ct_ntp = ct.get_probands_under(climb_up_step=climb_up_step)

    gg_prob = gg.probands(use_time=False)
    ct_prob = ct.probands(use_time=False)

    pairs = []
    scores = []

    for i, n_gg in enumerate(gg_nodes):
        for j, n_ct in enumerate(ct_nodes):

            if n_gg in gg_prob or n_ct in ct_prob:
                continue

            score = (len(gg_ntp[n_gg].intersection(ct_ntp[n_ct])) /
                     len(gg_ntp[n_gg].union(ct_ntp[n_ct])))

            pairs += [[n_ct, n_gg]]
            scores.append(score)

    return np.array(pairs).T, np.array(scores)

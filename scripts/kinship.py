from genealogy_aligner.kinship import kinship_matrix

balsac = Pedigree.read_balsac('data/balsac.tsv').to_table()

K = kinship_matrix(balsac.individual, balsac.mother, balsac.father, balsac.time)


print('Sparsifying')
from scipy.sparse import bsr_matrix, save_npz, load_npz


save_npz('cached/b140-kinship-sparse-bsr.npz', bsr_matrix(K))


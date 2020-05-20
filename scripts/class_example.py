# Example showing the class hierarchy

from genealogy_aligner import Pedigree
import matplotlib.pyplot as plt
from util import get_basename

founders = 10
generations = 4
avg_children = 2
avg_out_of_family = 2
G = Pedigree.simulate_from_founders(founders, generations, avg_children, avg_out_of_family)
    
T = G.sample_path()
C = T.to_coalescent_tree()

fig, ax = plt.subplots(ncols=3, figsize=(18, 6))

G.draw(ax=ax[0])
ax[0].set_title('Genealogy')
T.draw(ax=ax[1])
ax[1].set_title('Coalescent traversal')
C.draw(ax=ax[2])
ax[2].set_title('Coalescent tree')

fig.savefig(f'fig/{get_basename(__file__)}.svg', dpi=300)
fig.savefig(f'fig/{get_basename(__file__)}.png', dpi=300)



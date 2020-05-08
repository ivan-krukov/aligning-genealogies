from Genealogy import Genealogy
import matplotlib.pyplot as plt

families = 5
generations = 4
avg_children = 2
avg_out_of_family = 2
G = Genealogy.from_founders(families, generations, avg_children, avg_out_of_family)
while G.generations < generations:
    # if a simulation terminated early (everyone died)
    print('retrying')
    G = Genealogy.from_founders(families, generations, avg_children)
    
T = G.sample_path()
C = T.to_coalescent_tree()

fig, ax = plt.subplots(ncols=3, figsize=(18, 6))

G.draw(ax=ax[0])
ax[0].set_title('Genealogy')
T.draw(ax=ax[1])
ax[1].set_title('Coalescent traversal')
C.draw(ax=ax[2])
ax[2].set_title('Coalescent tree')

fig.savefig('fig/class_example.svg', dpi=300)


from Genealogy import Genealogy
import matplotlib.pyplot as plt

families = 5
generations = 7
avg_children = 2.2
G = Genealogy.from_founders(families, generations, avg_children)
while G.generations < generations:
    # if a simulation terminated early (everyone died)
    print('retrying')
    G = Genealogy.from_founders(families, generations, avg_children)
    
T = G.sample_path()

fig, ax = plt.subplots(ncols=2, figsize=(20, 10))

G.draw(ax=ax[0])
ax[0].set_title('Genealogy')
T.draw(ax=ax[1])
ax[1].set_title('Coalescent traversal')

fig.savefig('fig/class_example.png', dpi=300)


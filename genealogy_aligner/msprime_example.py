from genealogy_aligner import Pedigree

ped = Pedigree.from_founders(10, 10, 2, 1)
sim = ped.generate_msprime_simulations()

with open('../fig/1.svg', 'w') as out:
    print(sim.first().draw(), file=out)

from Pedigree import Pedigree

ped = Pedigree.simulate_from_founders(10, 10, 2, 1)
sim = ped.generate_msprime_simulations()


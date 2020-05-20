from genealogy_aligner import Pedigree

ped = Pedigree.simulate_from_founders(10, 10, 2, 1)
sim = ped.generate_msprime_simulations(complete=False)

balsac = Pedigree.read_balsac('data/balsac.tsv')
bsim = balsac.generate_msprime_simulations(complete=False)


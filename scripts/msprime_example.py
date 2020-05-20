from genealogy_aligner import Pedigree

ped = Pedigree.simulate_from_founders(10, 10, 2, 1)
probands = ped.probands(use_time=False)
sim = ped.generate_msprime_simulations(probands, complete=False)

balsac = Pedigree.read_balsac('data/balsac.tsv')
bprobands = balsac.probands(use_time=False)
bsim = balsac.generate_msprime_simulations(bprobands, complete=False)


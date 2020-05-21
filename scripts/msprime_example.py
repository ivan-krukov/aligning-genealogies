from genealogy_aligner import Pedigree

print("Simulating from a random pedigree...")
ped = Pedigree.simulate_from_founders(10, 10, 2, 1)
probands = ped.probands(use_time=False)
sim = ped.generate_msprime_simulations(model_after=None)

print("Simulating from BALSAC pedigree...")
balsac = Pedigree.from_table('../data/balsac.tsv', header=True)
bprobands = balsac.probands(use_time=False)
bsim = balsac.generate_msprime_simulations(model_after=None)

print("Done!")

from genealogy_aligner import Pedigree

print("Simulating from a random pedigree...")
ped = Pedigree.simulate_from_founders(10, 10, 2, 1)
probands = ped.probands(use_time=False)
sim = ped.generate_msprime_simulations(model_after=None)

print("Simulating from geneaJi pedigree...")
gen_ji = Pedigree.from_table('../data/geneaJi.tsv', header=True, check_2_parents=False)
print(gen_ji.attributes)
gji_sim = gen_ji.generate_msprime_simulations(model_after=None)

print("Simulating from a sample pedigree (kinship2)...")
sample_ped = Pedigree.from_table('../data/kinship2_sample1_ped.tsv', header=True)
print(sample_ped.attributes)
sp_sim = sample_ped.generate_msprime_simulations(model_after=None)

print("Simulating from BALSAC pedigree...")
balsac = Pedigree.from_table('../data/balsac.tsv', header=True)
bprobands = balsac.probands(use_time=False)
bsim = balsac.generate_msprime_simulations(model_after=None)

print("Done!")

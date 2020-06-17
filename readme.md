# Genome Imputation by Aligning Coalescent Trees with Genealogies

## Development

Please run the tests in `run-tests.sh` before committing. Add this script to your pre-commit hook:

```sh
cd .git/hooks
ln -s ../../run-tests.sh ./pre-commit
```

## Dependencies:

To access the relevant `msprime` functionalities, please install:

```
pip install --user newick
pip install --user git+https://github.com/DomNelson/msprime.git@pedigree_ind_fix
```

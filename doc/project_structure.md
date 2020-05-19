# Package structure

```
aligning_genealogies
├── cached/ ........................ Temporary data - no git
├── doc/ ........................... Documentation, notes
├── fig/ ........................... Figures
├── data/   ........................ Permanent data
├── genealogy_aligner/ ............. Package source
├── test/ .......................... Package tests
├── examples/ ...................... Example scripts
│   └── 'genealogy_aligner/ ........ Symlink to package source
│   └── util.py .................... Utilities for scripts
└── notebooks/ ..................... Jupyter notebooks
    └── 'genealogy_aligner/ ........ Symlink to package source 
```

# Python imports

Imports from `notebooks` and `examples` are like `from genealogy_aligner.Pedigree import _`

Imports froh the package are like `from .Pedigree import _`

# Figures

Every file in `fig` has a python script with the same name. 
Multiple extensions (_e.g._ `.pdf`, `.png`) are okay.

```python
from util import get_basename
# -- snip --
fig.savefig(f'fig/{get_basename(__file__)}.svg', dpi=300)
fig.savefig(f'fig/{get_basename(__file__)}.png', dpi=300)
```

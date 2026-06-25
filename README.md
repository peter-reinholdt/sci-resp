# LR-SCI(-PT)

This repository contains code and example scripts for a linear response selected configuration interaction (LR-SCI) implementation, described in https://arxiv.org/abs/2510.02949.

We provide example scripts for computing polarizabilities (`polarizability.py`), damped response `damped-response.py`, and spin-spin coupling constant calculations `sscc.py`.

Perturbatively corrected variants are available for polarizabilities (`polarizability_pt.py`), damped response `damped-response_pt.py`, and spin-spin coupling constant calculations `sscc_pt.py`,
but note that the PT-corrected damped response does not correct the pole structure (so don't use it).


# Requirements
- Numpy
- PySCF (https://github.com/pyscf/pyscf)
- PyCI (https://github.com/theochem/PyCI). The perturbative corrections use some extra routines available at [https://github.com/peter-reinholdt/PyCI](https://github.com/peter-reinholdt/PyCI/tree/V-matvec)

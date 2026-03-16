#!/usr/bin/env python

import argparse
import pyscf
import numpy as np
import pyci
from solvers import solve_ci
from response import resp, resp_pt2, wrap_matvec, one_electron_ao2mo

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', type=str, required=True)
parser.add_argument('--basis', type=str, required=True)
parser.add_argument('--ncore', type=int, default=0)
parser.add_argument('--couple-property', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--couple-response', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--eps2', type=float, default=1e-9)
parser.add_argument('--natorb', action='store_true')
args = parser.parse_args()

xyz = args.xyz
basis = args.basis

m = pyscf.M(atom=xyz, basis=basis, symmetry=True)
ncore = args.ncore
ncas = m.nao - args.ncore
nelcas = sum(m.nelec) - 2*ncore
nbeta = nelcas // 2
nalpha = nelcas - nbeta
nelec = (nalpha, nbeta)
print(f'{nelec=} in {ncas=}')

m.max_memory = 3000 # 3 GB
mf = pyscf.scf.RHF(m).run()
mf.conv_tol = 1e-12
mf.kernel()
cas = pyscf.mcscf.CASCI(mf, ncas, nelcas)

h1, ecore = cas.h1e_for_cas()
eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
ham = pyci.hamiltonian(ecore, h1, eri.transpose(0,2,1,3))

wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
wfn.add_hartreefock_det()
dets_added = 1
op = pyci.sparse_op(ham, wfn)
e_vecs = np.array([[1.]])
e_vals = op.get_element(0,0) + op.ecore
old_energy = np.min(e_vals)
niter = 0

# 1) Solve for |Psi_0>
eps = args.eps
eps2 = args.eps2
eps_mu = eps if args.couple_property else None
eps_resp = eps if args.couple_response else None

dets_added = True
while dets_added:
    # Add connected determinants to wave function via HCI
    dets_added = pyci.add_hci(ham, wfn, e_vecs[:, 0], eps=eps)
    # Update CI matrix operator
    op.update(ham, wfn)
    # Solve CI matrix problem
    e_vecs = np.concatenate([e_vecs, np.zeros((dets_added, e_vecs.shape[1]))], axis=0)
    matvec = lambda v: wrap_matvec(op, v)
    hdiag = op.diagonal()
    e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
    e_vals += op.ecore
    delta_e = old_energy - np.min(e_vals)
    old_energy = np.min(e_vals)
    niter += 1
    num_determinants = e_vecs.shape[0]
    print(f'{niter=} {eps=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')

if args.natorb:
    print("Forming natural orbitals and reforming SCI ground state wave function...")
    d1 = pyci.compute_rdm1(wfn, e_vecs[0])
    rdm1 = d1[0] + d1[1]
    rdm1 = _make_rdm1_on_mo(rdm1, ncore, ncas, m.nao)
    noons, natorbs = np.linalg.eigh(rdm1)
    noons = np.flip(noons)
    print('Natural occupation numbers:', noons)
    natorbs = np.flip(natorbs, axis=1)

    mf.mo_coeff = mf.mo_coeff @ natorbs
    cas = pyscf.mcscf.CASCI(mf, ncas, nelcas)
    h1, ecore = cas.h1e_for_cas()
    eri = pyscf.ao2mo.full(m, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
    ham = pyci.hamiltonian(ecore, h1, eri.transpose(0,2,1,3))
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    wfn.add_hartreefock_det()
    dets_added = 1
    op = pyci.sparse_op(ham, wfn)
    e_vecs = np.array([[1.]])
    e_vals = op.get_element(0,0) + op.ecore
    old_energy = np.min(e_vals)
    niter = 0

    dets_added = True
    while dets_added:
        # Add connected determinants to wave function via HCI
        dets_added = pyci.add_hci(ham, wfn, e_vecs[:,0], eps=eps)
        # Update CI matrix operator
        op.update(ham, wfn)
        # Solve CI matrix problem
        e_vecs = np.concatenate([e_vecs, np.zeros((dets_added, e_vecs.shape[1]))], axis=0)
        matvec = lambda v: wrap_matvec(op, v)
        hdiag = op.diagonal()
        e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
        e_vals += op.ecore
        delta_e = old_energy - np.min(e_vals)
        old_energy = np.min(e_vals)
        niter += 1
        num_determinants = e_vecs.shape[0]
        print(f'{niter=} {eps=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')

dipole_integrals_ao = m.intor('int1e_r')
dipole_integrals_mo = [one_electron_ao2mo(cas, integral) for integral in dipole_integrals_ao]
labels = ['X', 'Y', 'Z']
integrals  = {label: integral for (label, integral) in zip(labels, dipole_integrals_mo)}
perturbations = [(label, frequency, parity) for label in labels 
                                            for frequency in [0.0] 
                                            for parity in [0]]

resp_pt2(ham, wfn, op, e_vecs, integrals, perturbations, eps2, eps_mu=eps_mu, eps_resp=eps_resp)

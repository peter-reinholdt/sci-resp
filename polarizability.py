#!/usr/bin/env python

import argparse
import pyscf
import numpy as np
import pyci
from solvers import solve_ci
from response import resp, wrap_matvec, one_electron_ao2mo, print_var_summary
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', type=str, required=True)
parser.add_argument('--basis', type=str, required=True)
parser.add_argument('--ncore', type=int, default=0)
parser.add_argument('--couple-property', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--couple-response', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--component', type=str, default=None)
parser.add_argument('--state', type=int, default=0)
parser.add_argument('--load', type=str)
args = parser.parse_args()

xyz = args.xyz
basis = args.basis

mol = pyscf.M(atom=xyz, basis=basis, symmetry=True)
ncore = args.ncore
ncas = mol.nao - args.ncore
nelcas = sum(mol.nelec) - 2*ncore
nbeta = nelcas // 2
nalpha = nelcas - nbeta
nelec = (nalpha, nbeta)
print(f'{nelec=} in {ncas=}')

mol.max_memory = 3000 # 3 GB
mf = pyscf.scf.RHF(mol).run()
mf.conv_tol = 1e-12
mf.kernel()
reference_state = None
if args.load:
    with h5py.File(args.load, 'r') as f:
        mf.mo_coeff = f['mo_coeff'][()]

cas = pyscf.mcscf.CASCI(mf, ncas, nelcas)

h1, ecore = cas.h1e_for_cas()
eri = pyscf.ao2mo.full(mol, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
ham = pyci.hamiltonian(ecore, h1, eri.transpose(0,2,1,3))

wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
wfn.add_hartreefock_det()
state = args.state
nroots = state + 1

wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
if args.load:
    with h5py.File(args.load, 'r') as f:
        for det in f['dets']:
            wfn.add_det(det)
        e_vecs = f['civec'][()]
        reference_state = e_vecs[:, state]
        op = pyci.sparse_op(ham, wfn)
        matvec = lambda v: wrap_matvec(op, v)
        hdiag = op.diagonal()
        e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
        assert converged
        old_energy = e_vals[state]
else:
    wfn.add_hartreefock_det()
    if nroots > 1:
        wfn.add_excited_dets(1)
    dets_added = len(wfn)
    op = pyci.sparse_op(ham, wfn)
    e_vals, e_vecs = op.solve(n=nroots)
    e_vecs = e_vecs.T
    old_energy = e_vals[state]

# 1) Solve for |Psi_0>
niter = 0
eps = args.eps
eps_mu = eps if args.couple_property else None
eps_resp = eps if args.couple_response else None

dets_added = True
while dets_added:
    # Add connected determinants to wave function via HCI
    screen_vector = np.max(np.abs(e_vecs), axis=1)
    dets_added = pyci.add_hci(ham, wfn, screen_vector, eps=eps)
    # Update CI matrix operator
    op.update(ham, wfn)
    # Solve CI matrix problem
    e_vecs = np.concatenate([e_vecs, np.zeros((dets_added, e_vecs.shape[1]))], axis=0)
    matvec = lambda v: wrap_matvec(op, v)
    hdiag = op.diagonal()
    e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
    while not converged:
        e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
    e_vals += op.ecore
    delta_e = old_energy - e_vals[state]
    old_energy = e_vals[state]
    niter += 1
    num_determinants = e_vecs.shape[0]
    for root in range(nroots):
        istarget = ' * ' if root == state else '   '
        ΔE_au = e_vals[root] - e_vals[0]
        ΔE_eV = (e_vals[root] - e_vals[0]) * 27.211396
        print(f'{root=}{istarget} {e_vals[root]=: 16.12f} {ΔE_au=: 12.8f} {ΔE_eV=: 12.8f}')
    print(f'{niter=} {eps=} {e_vals[state]=} {delta_e=} {dets_added=} {num_determinants=}')

if args.component is not None:
    component = args.component.upper()
    index = {'X': 0, 'Y': 1, 'Z': 2}[component]
    dipole_integrals_ao = [mol.intor('int1e_r')[index]]
    labels = [component]
else:
    dipole_integrals_ao = mol.intor('int1e_r')
    labels = ['X', 'Y', 'Z']

dipole_integrals_mo = [one_electron_ao2mo(cas, integral) for integral in dipole_integrals_ao]
integrals  = {label: integral for (label, integral) in zip(labels, dipole_integrals_mo)}
perturbations = [(label, frequency, parity) for label in labels 
                                            for frequency in [0.0] 
                                            for parity in [0]]

wfn, op, e_vecs, response_vectors, property_vectors, response_functions = resp(ham, wfn, op, e_vecs, integrals, perturbations, eps_mu=eps_mu, eps_resp=eps_resp, state=state, reference_state=reference_state)
print_var_summary(response_functions)

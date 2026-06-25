#!/usr/bin/env python

import argparse
import pyscf
import numpy as np
import pyci
from solvers import solve_ci
from response import resp, wrap_matvec, one_electron_ao2mo, print_var_summary, _make_rdm1_on_mo
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', type=str, required=True)
parser.add_argument('--basis', type=str, required=True)
parser.add_argument('--ncore', type=int, default=0)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--component', type=str, default=None)
parser.add_argument('--state', type=int, default=0)
parser.add_argument('--save', type=str)
parser.add_argument('--load', type=str)
parser.add_argument('--natorb', action='store_true')
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
if args.load:
    with h5py.File(args.load, 'r') as f:
        mf.mo_coeff = f['mo_coeff'][()]

cas = pyscf.mcscf.CASCI(mf, ncas, nelcas)

h1, ecore = cas.h1e_for_cas()
eri = pyscf.ao2mo.full(mol, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
ham = pyci.hamiltonian(ecore, h1, eri.transpose(0,2,1,3))

state = args.state
nroots = state + 1

wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
if args.load:
    with h5py.File(args.load, 'r') as f:
        for det in f['dets']:
            wfn.add_det(det)
        e_vecs = f['civec'][()]
        reference_state = e_vecs[:, -1]
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
    reference_state = None

# 1) Solve for |Psi_0>
niter = 0
eps = args.eps

dets_added = True
while dets_added:
    # Add connected determinants to wave function via HCI
    # screen_vector = np.max(np.abs(e_vecs), axis=1)
    # dets_added = pyci.add_hci(ham, wfn, screen_vector, eps=eps)
    dets_added = pyci.add_hci(ham, wfn, e_vecs[:, state], eps=eps)
    # Update CI matrix operator
    op.update(ham, wfn)
    # Solve CI matrix problem
    e_vecs = np.concatenate([e_vecs, np.zeros((dets_added, e_vecs.shape[1]))], axis=0)
    matvec = lambda v: wrap_matvec(op, v)
    hdiag = op.diagonal()
    e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
    while not converged:
        e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
    nroots = len(e_vals)
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

if args.natorb:
    print("Forming natural orbitals and reforming SCI ground state wave function...")
    d1 = pyci.compute_rdm1(wfn, e_vecs[:, state])
    rdm1 = d1[0] + d1[1]
    rdm1 = _make_rdm1_on_mo(rdm1, ncore, ncas, mol.nao)
    noons, natorbs = np.linalg.eigh(rdm1)
    noons = np.flip(noons)
    print('Natural occupation numbers:', noons)
    natorbs = np.flip(natorbs, axis=1)

    mf.mo_coeff = mf.mo_coeff @ natorbs
    cas = pyscf.mcscf.CASCI(mf, ncas, nelcas)
    h1, ecore = cas.h1e_for_cas()
    eri = pyscf.ao2mo.full(mol, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
    ham = pyci.hamiltonian(ecore, h1, eri.transpose(0,2,1,3))
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    wfn.add_hartreefock_det()
    if nroots > 1:
        wfn.add_excited_dets(1)
    dets_added = len(wfn)
    op = pyci.sparse_op(ham, wfn)
    e_vals, e_vecs = op.solve(n=nroots)
    e_vecs = e_vecs.T
    delta_e = old_energy - e_vals[state]
    old_energy = e_vals[state]
    niter = 0

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
        e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True)
        while not converged:
            e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True)
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


if args.save:
    with h5py.File(args.save, 'w') as f:
        f['mo_coeff'] = mf.mo_coeff
        f['dets'] = wfn.to_det_array()
        f['civec'] = e_vecs

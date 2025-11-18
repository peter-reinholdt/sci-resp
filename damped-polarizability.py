#!/usr/bin/env python

import argparse
import pyscf
import numpy as np
import pyci
from solvers import solve_ci
from response import resp, wrap_matvec

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', type=str, required=True)
parser.add_argument('--basis', type=str, required=True)
parser.add_argument('--ncore', type=int, default=0)
parser.add_argument('--couple-property', type=bool, default=True)
parser.add_argument('--couple-response', type=bool, default=True)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--freq', type=float, required=True)
parser.add_argument('--gamma', type=float, default=0.01469972198)
args = parser.parse_args()

xyz = args.xyz
basis = args.basis

m = pyscf.M(atom=xyz, basis=basis, symmetry=True)
ncore = args.ncore
ncas = m.nao
nelcas = sum(m.nelec)
nelec = m.nelec
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
eps_mu = eps
eps_resp = eps

dets_added = True
while dets_added:
    # Add connected determinants to wave function via HCI
    dets_added = pyci.add_hci(ham, wfn, e_vecs[0], eps=eps)
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


integrals = m.intor('int1e_r')
labels = ['XX', 'YY', 'ZZ']

gamma = args.gamma
omega = args.freq

res = []
ims = []
for operator, label in zip(integrals, labels):
    wfn_plus, e_vecs_plus, x_plus, zpsi_plus = resp(cas, ham, nelec, wfn, e_vecs, operator, omega, gamma, eps_mu, eps_resp, couple_property=args.couple_property, couple_response=args.couple_response)
    wfn_minus, e_vecs_minus, x_minus, zpsi_minus = resp(cas, ham, nelec, wfn, e_vecs, -operator, -omega, -gamma, eps_mu, eps_resp, couple_property=args.couple_property, couple_response=args.couple_response)
    re = np.dot(x_plus.real, zpsi_plus) + np.dot(x_minus.real, zpsi_minus)
    im = np.dot(x_plus.imag, zpsi_plus) + np.dot(x_minus.imag, zpsi_minus)
    res.append(re)
    ims.append(im)
    print(f'{label=} {omega=} {re=} {im=} {len(x_plus)=} {len(x_minus)=}', flush=True)
re_average = np.average(res)
im_average = np.average(ims)
print(f'Average {omega=} {re_average=} {im_average=}')

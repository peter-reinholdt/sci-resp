#!/usr/bin/env python

import sys
import time
import argparse
import pyscf
import numpy as np
import pyci
from functools import reduce
from solvers import davidson_response, solve_ci
from one_electron_operator import singlet_operator_state, triplet_operator_state
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--couple-property', type=bool, default=True)
parser.add_argument('--couple-response', type=bool, default=True)
parser.add_argument('--eps', type=float, default=1e-3)
parser.add_argument('--freq', type=float, required=True)
args = parser.parse_args()


def one_electron_operator_for_cas(casci, operator, mo_coeff=None, ncas=None, ncore=None):
    # based on the pyscf thing
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    mo_cas = mo_coeff[:,ncore:ncore+ncas]
    h1 = reduce(np.dot, (mo_cas.conj().T, operator, mo_cas))
    return h1

def _make_rdm1_on_mo(casdm1, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = np.zeros((nmo,nmo))
    idx = np.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1
    return dm1

def wrap_matvec(op, x):
    if x.ndim == 1:
        return op.matvec(x)
    elif x.ndim == 2:
        out = np.zeros_like(x)
        for i in range(x.shape[1]):
            out[:,i] = op.matvec(x[:,i])
        return out


xyz = 'water.xyz'
basis = 'cc-pVDZ'
m = pyscf.M(atom=xyz, basis=basis, symmetry=True)
ncore = 0
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


def resp(ham, nelec, gs_wfn, gs_evecs, integral, omega, gamma, couple_property=True, couple_response=True, triplet=False):
    property_operator = triplet_operator_state if triplet else singlet_operator_state
    
    # setup common stuff
    wfn = pyci.fullci_wfn(ham.nbasis, *nelec)
    wfn.add_dets_from_wfn(gs_wfn)
    op = pyci.sparse_op(ham, wfn)
    op.update(ham, wfn)
    matvec = lambda v: wrap_matvec(op, v)
    hdiag = op.diagonal()
    e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=None, verbose=True)
    e_vals += op.ecore
    old_energy = np.min(e_vals)
    mu =  one_electron_operator_for_cas(cas, integral)
    ham_mu = pyci.hamiltonian(0., mu, eri*0)

    if couple_property:
        # Add determinants connecting via the one-electron operator, and re-solve
        dets_added = pyci.add_hci(ham_mu, wfn, e_vecs[0], eps=eps_mu)
        op.update(ham, wfn)
        e_vecs = np.concatenate((e_vecs, np.zeros((dets_added, e_vecs.shape[1]))), axis=0)
        matvec = lambda v: wrap_matvec(op, v)
        hdiag = op.diagonal()
        e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
        e_vals += op.ecore
        delta_e = old_energy - np.min(e_vals)
        old_energy = np.min(e_vals)
        num_determinants = e_vecs.shape[0]
        print(f'couple_property {eps=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')

    # Build z|Psi0>
    #mu_op = pyci.sparse_op(ham_mu, wfn)
    #zPsi = mu_op.matvec(e_vecs)
    zPsi = property_operator(wfn.to_det_array(), mu, e_vecs[:,0])
    zPsi -= e_vecs[:,0] * np.dot(e_vecs[:,0], zPsi)

    E0 = np.dot(e_vecs[:,0], matvec(e_vecs[:,0]))
    # 4) Solve response equation
    E0w = E0 + omega + 1j * gamma
    x = davidson_response(lambda v: matvec(v)-E0w*v, zPsi, hdiag-E0w, verbose=False)

    # resolve with more added
    if couple_response:
        while True:
            dets_added = pyci.add_hci(ham, wfn, x, eps=eps_resp)
            op.update(ham, wfn)
            e_vecs = np.concatenate((e_vecs, np.zeros((dets_added, e_vecs.shape[1]))), axis=0)
            matvec = lambda v: wrap_matvec(op, v)
            hdiag = op.diagonal()
            e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
            e_vals += op.ecore
            delta_e = old_energy - np.min(e_vals)
            old_energy = np.min(e_vals)
            num_determinants = e_vecs.shape[0]
            zPsi = property_operator(wfn.to_det_array(), mu, e_vecs[:,0])
            zPsi -= e_vecs[:,0] * np.dot(e_vecs[:,0], zPsi)
            E0 = np.dot(e_vecs[:,0], matvec(e_vecs[:,0]))
            E0w = E0 + omega + 1j * gamma
            x0 = np.concatenate((x, np.zeros(dets_added)))
            x = davidson_response(lambda v: matvec(v)-E0w*v, zPsi, hdiag-E0w, verbose=False, guess=x0)
            if dets_added / num_determinants < 0.01:
                print(f'finised_couple_response {eps=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')
                break
            else:
                print(f'couple_response {eps=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')
    return wfn, e_vecs, x, zPsi

gamma = 0.01469972198

integrals = m.intor('int1e_r')
labels = ['XX', 'YY', 'ZZ']

omega = args.freq
res = []
ims = []
for operator, label in zip(integrals, labels):
    wfn_plus, e_vecs_plus, x_plus, zpsi_plus = resp(ham, nelec, wfn, e_vecs, operator, omega, gamma, couple_property=args.couple_property, couple_response=args.couple_response)
    wfn_minus, e_vecs_minus, x_minus, zpsi_minus = resp(ham, nelec, wfn, e_vecs, -operator, -omega, gamma, couple_property=args.couple_property, couple_response=args.couple_response)
    re = np.dot(x_plus.real, zpsi_plus) - np.dot(x_minus.real, zpsi_minus)
    im = np.dot(x_plus.imag, zpsi_plus) + np.dot(x_minus.imag, zpsi_minus)
    res.append(re)
    ims.append(im)
    print(f'{label=} {omega=} {re=} {im=} {len(x_plus)=} {len(x_minus)=}', flush=True)
re_average = np.average(res)
im_average = np.average(ims)
print(f'Average {omega=} {re_average=} {im_average=}')

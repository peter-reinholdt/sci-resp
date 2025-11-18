#!/usr/bin/env python

import pyci
import numpy as np
from solvers import davidson_response, solve_ci
from one_electron_operator import singlet_operator_state, triplet_operator_state
from functools import reduce

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

def response_dot(cas, response_wfn, response_vector, response_e_vecs, h1ao, triplet=None):
    #             ~
    # evaluate <X|B>; form property vector B from the determinant set used in response vector X
    property_operator = triplet_operator_state if triplet else singlet_operator_state
    mu =  one_electron_operator_for_cas(cas, h1ao)
    property_vector = property_operator(response_wfn.to_det_array(), mu, response_e_vecs.ravel())
    property_vector -= response_e_vecs.ravel() * np.dot(property_vector, response_e_vecs.ravel())
    return np.dot(response_vector, property_vector)

def resp(cas, ham, nelec, gs_wfn, gs_evecs, integral, omega, gamma, eps_mu, eps_resp, couple_property=True, couple_response=True, triplet=False):
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
    ham_mu = pyci.hamiltonian(0., mu, np.zeros_like(ham.two_mo))

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
        print(f'couple_property {eps_mu=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')

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
            dets_added = pyci.add_hci(ham, wfn, np.abs(x), eps=eps_resp)
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
                print(f'finised_couple_response {eps_resp=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')
                break
            else:
                print(f'couple_response {eps_resp=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')
    return wfn, e_vecs, x, zPsi

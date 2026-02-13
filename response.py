#!/usr/bin/env python

import pyci
import numpy as np
from solvers import davidson_response, solve_ci
from functools import reduce
from collections import defaultdict
import time

def one_electron_ao2mo(casci, operator, mo_coeff=None, ncas=None, ncore=None):
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

def zeropad(x, N):
    shape = (N,) + x.shape[1:]
    out = np.zeros(shape, dtype=x.dtype)
    out[:len(x), ...] = x
    return out

def resp(ham, wfn, op, e_vecs, integrals, perturbations, frequency=0.0, gamma=0.0, triplet=False, eps_mu=None, eps_resp=None):
    """
    Args:
        ham (pyci.Hamiltonian): electronic hamiltonian
        wfn (pyci.fullci_wfn): pyci wave function object - (note: this is modified when couple_property and/or couple_response is True)
        op (pyci.sparse_op):  sparse operator object for the electronic hamiltonian
        e_vecs (np.ndarray): (ground-)state coefficient vector
        integrals (dict: label->np.ndarray): MO integrals (nmo, nmo)
        perturbations:
        frequency (float): frequency of perturbation
        gamma (float): damping factor for damped response
        couple_property (bool): Activate coupling to property vector
        couple_response (bool): Activate coupling to response vector
        triplet (bool): Is the one-electron operator triplet (otherwise singlet)
    
    Returns:
    """
    couple_property = eps_mu is not None
    couple_response = eps_resp is not None
    matvec = lambda v: wrap_matvec(op, v)
    hdiag = op.diagonal()
    e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
    e_vals += op.ecore
    old_energy = np.min(e_vals)
    ham_perturbations = {label: pyci.hamiltonian(0., integral, ham.two_mo*0) for (label, integral) in integrals.items()}

    # for each perturbation, add determinants connecting via the one-electron operator, and re-solve
    if couple_property:
        dets_added = 0
        for ham_perturbation in ham_perturbations.values():
            dets_added += pyci.add_hci(ham_perturbation, wfn, e_vecs[:, 0], eps=eps_mu)
            num_determinants = len(wfn.to_det_array())
            e_vecs = zeropad(e_vecs, num_determinants)
        op.update(ham, wfn)
        matvec = lambda v: wrap_matvec(op, v)
        hdiag = op.diagonal()
        e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
        e_vals += op.ecore
        delta_e = old_energy - np.min(e_vals)
        old_energy = np.min(e_vals)
        print(f'couple_property {eps_mu=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')

    # form property vectors
    property_vectors = {}
    for (label, omega, parity) in perturbations:
        if label in property_vectors:
            continue
        ham_perturbation = ham_perturbations[label]
        property_vector = ham_perturbation.one_electron_direct(wfn, e_vecs[:,0], triplet=triplet)
        property_vector -= e_vecs[:,0] * np.dot(e_vecs[:,0], property_vector)
        property_vectors[label] = property_vector

    # solve response equations
    E0 = np.dot(e_vecs[:,0], matvec(e_vecs[:,0]))
    response_vectors = {}
    for (label, omega, parity) in perturbations:
        E0w = E0 + parity * (omega + 1j * gamma)
        property_vector = parity*property_vectors[label]
        response_vector = davidson_response(lambda v: matvec(v)-E0w*v, property_vector, hdiag-E0w, verbose=False)
        response_vectors[(label, omega, parity)] = response_vector

    # add determinants coupling through response vector (and solve response equations again) 
    if couple_response:
        while True:
            # add determinants
            dets_added = 0
            for (label, omega, parity), response_vector in response_vectors.items():
                num_determinants = len(wfn.to_det_array())
                response_vector = zeropad(response_vector, num_determinants)
                dets_added += pyci.add_hci(ham, wfn, response_vector.real, eps=eps_resp)
                if np.any(response_vector.imag) != 0.:
                    dets_added += pyci.add_hci(ham, wfn, response_vector.imag, eps=eps_resp)
            num_determinants = len(wfn.to_det_array())
            e_vecs = zeropad(e_vecs, num_determinants)
            op.update(ham, wfn)
            matvec = lambda v: wrap_matvec(op, v)
            hdiag = op.diagonal()
            e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
            e_vals += op.ecore
            delta_e = old_energy - np.min(e_vals)
            old_energy = np.min(e_vals)
            # form property vectors
            property_vectors = {}
            for (label, omega, parity) in perturbations:
                if label in property_vectors:
                    continue
                ham_perturbation = ham_perturbations[label]
                property_vector = ham_perturbation.one_electron_direct(wfn, e_vecs[:,0], triplet=triplet)
                property_vector -= e_vecs[:,0] * np.dot(e_vecs[:,0], property_vector)
                property_vectors[label] = property_vector
            # solve response equations
            E0 = np.dot(e_vecs[:,0], matvec(e_vecs[:,0]))
            for (label, omega, parity) in perturbations:
                E0w = E0 + parity * (omega + 1j * gamma)
                property_vector = parity*property_vectors[label]
                response_vector = davidson_response(lambda v: matvec(v)-E0w*v, property_vector, hdiag-E0w, verbose=False)
                response_vectors[(label, omega, parity)] = response_vector
            # exit when almost no new determinants are added
            if dets_added / num_determinants < 0.01:
                print(f'finised_couple_response {eps_resp=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')
                break
            else:
                print(f'couple_response {eps_resp=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')
    return wfn, op, e_vecs, response_vectors, property_vectors


def print_pt2_summary(response_functions):
    labels = [key for key in response_functions.keys() if key[-1] == 2]
    for label in labels:
        label1, label2, omega, _ = label
        Δ_0 = response_functions[(label1, label2, omega, 0)]
        Δ_1 = response_functions[(label1, label2, omega, 1)]
        Δ_2 = response_functions[(label1, label2, omega, 2)]
        Δ_0_int = response_functions[(label1, label2, omega, 0, 'internal')]
        value = Δ_0 + Δ_1 + Δ_2
        ω = omega
        if value.imag != 0:
            # complex
            imag = value.imag, Δ_0.imag, Δ_1.imag, Δ_2.imag, Δ_0_int.imag
            value = value.real
            Δ_0 = Δ_0.real
            Δ_1 = Δ_1.real 
            Δ_2 = Δ_2.real
            Δ_0_int = Δ_0_int.real
            print(f'Re<<{label1}; {label2}>>({ω=:}) {value=:16.9f} {Δ_0=:16.9f} {Δ_1=:16.9f} {Δ_2=:16.9f} {Δ_0_int=:16.9f}')
            value, Δ_0, Δ_1, Δ_2, Δ_0_int = imag
            print(f'Im<<{label1}; {label2}>>({ω=:}) {value=:16.9f} {Δ_0=:16.9f} {Δ_1=:16.9f} {Δ_2=:16.9f} {Δ_0_int=:16.9f}')
        else:
            # real
            print(f'<<{label1}; {label2}>>({ω=:}) {value=:16.9f} {Δ_0=:16.9f} {Δ_1=:16.9f} {Δ_2=:16.9f} {Δ_0_int=:16.9f}')


def resp_pt2(ham, wfn, op, e_vecs, integrals, perturbations, eps2, frequency=0.0, gamma=0.0, triplet=False, eps_mu=None, eps_resp=None):
    # solve the internal/variational LR problem
    # (we run GS, GS+V, GS+X, or GS+V+X)
    print('LR-SCI-PT solving variational equations...')
    t1 = time.time()
    wfn, op, e_vecs, response_vectors, property_vectors = resp(ham, wfn, op, e_vecs, integrals, perturbations, frequency=frequency, gamma=gamma, triplet=triplet, eps_mu=eps_mu, eps_resp=eps_resp)
    t2 = time.time()
    print('LR-SCI-PT solving variational equations... done in', t2 - t1, 's')
    c0 = e_vecs.ravel()
    E0 = np.dot(c0, op.matvec(c0))

    N_internal = op.shape[0]
    def matvec(v, c0=c0):
        v = v - c0*np.dot(c0, v)
        Hv = op.matvec(v)
        Hv = Hv - c0 * np.dot(c0, Hv)
        return Hv

    # select perturbative space
    print(f'LR-SCI-PT det-add...')
    t1 = time.time()
    # from Hamiltonian
    dets_added = pyci.add_hci(ham, wfn, c0, eps2)
    print('Added', dets_added, 'from Hamiltonian')
    # from property operator
    ham_perturbations = {label: pyci.hamiltonian(0., integral, ham.two_mo*0) for (label, integral) in integrals.items()}
    for (label, ham_perturbation) in ham_perturbations.items():
        added = pyci.add_hci(ham_perturbation, wfn, c0, eps=eps2)
        print('Added', added, 'from operator', label)
        dets_added += added
    for (label, omega, parity), response_vector in response_vectors.items():
        added = pyci.add_hci(ham, wfn, response_vector.real, eps=eps2)
        if np.any(response_vector.imag != 0):
            added += pyci.add_hci(ham, wfn, response_vector.imag, eps=eps2)
        print('Added', added, 'from response vector', (label, omega, parity))
        dets_added += added
    t2 = time.time()
    print('LR-SCI-PT det-add... done in:  ', t2 - t1, 's')
    
    t1 = time.time()
    op.update_diagonal(ham, wfn)
    diagonal = op.diagonal()
    N_total = len(diagonal)
    N_external = N_total - N_internal
    t2 = time.time()
    print('LR-SCI-PT diagonal... done in: ', t2 - t1, 's')
    
    print(f'LR-SCI-PT epsilons  : {eps2=} {eps_mu=} {eps_resp=}')
    print(f'LR-SCI-PT dimensions: {N_internal=} {N_external=} {N_total=}')

    # compute first-order wave function coefficients (c1)
    print('LR-SCI-PT V@c0...')
    t1 = time.time()
    Vc0 = op.Vmatvec_direct(ham, wfn, N_internal, eps2, c0)
    print('LR-SCI-PT forming c1...')
    c1 = Vc0 / (E0 - diagonal)
    E2 = np.dot(c1, Vc0)
    t2 = time.time()
    print('LR-SCI-PT V@c0... done in:     ', t2 - t1, 's')
    
    # compute second-order wave function coefficients (c2)
    print('LR-SCI-PT V@c1...')
    t1 = time.time()
    rhs = -op.Vmatvec_direct(ham, wfn, N_internal, eps2, c1)
    t2 = time.time()
    print('LR-SCI-PT V@c1... done in:     ', t2 - t1, 's')
    print('LR-SCI-PT forming c2...')
    t1 = time.time()
    c2 = np.zeros_like(c1)
    rhs[:N_internal] += E2 * c0
    c2[N_internal:] = rhs[N_internal:] / (diagonal[N_internal:] - E0)
    t2 = time.time()
    c2[:N_internal] = davidson_response(lambda v: matvec(v) - E0*v, rhs[:N_internal], diagonal[:N_internal]-E0, verbose=True)
    t2 = time.time()
    print('LR-SCI-PT forming c2... done in', t2 - t1, 's')


    # compute zeroth, first, and second-order one-electron operator products
    # and form zeroth, first, and second-order property vectors
    property_vectors = {}
    moments = {}
    t1 = time.time()
    print('LR-SCI-PT compute property vectors...')
    for (label, omega, parity) in perturbations:
        if label in property_vectors:
            continue
        ham_perturbation = ham_perturbations[label]
        # zeroth order
        print(f'{label=}, order=0')
        product = ham_perturbation.one_electron_direct(wfn, c0, triplet=triplet)
        moments[(label, 0, 0)] = np.dot(c0, product[:N_internal])
        moments[(label, 1, 0)] = np.dot(c1, product)
        moments[(label, 2, 0)] = np.dot(c2, product)
        property_vectors[(label, 0)] = product
        property_vectors[(label, 0)][:N_internal] -= c0 * moments[(label, 0, 0)]

        # first order
        print(f'{label=}, order=1')
        product = ham_perturbation.one_electron_direct(wfn, c1, triplet=triplet)
        moments[(label, 0, 1)] = np.dot(c0, product[:N_internal])
        moments[(label, 1, 1)] = np.dot(c1, product)
        moments[(label, 2, 1)] = np.dot(c2, product)
        property_vectors[(label, 1)] = product
        property_vectors[(label, 1)][:N_internal] -= c0 * (moments[(label, 0, 1)] + moments[(label, 1, 0)])
        property_vectors[(label, 1)] -= c1 * (moments[(label, 0, 0)])

        # second order
        print(f'{label=}, order=2')
        product = ham_perturbation.one_electron_direct(wfn, c2, triplet=triplet)
        moments[(label, 0, 2)] = np.dot(c0, product[:N_internal])
        moments[(label, 1, 2)] = np.dot(c1, product)
        moments[(label, 2, 2)] = np.dot(c2, product)
        property_vectors[(label, 2)] = product
        property_vectors[(label, 2)][:N_internal] -= c0 * (moments[(label, 0, 2)] + moments[(label, 1, 1)] + moments[(label, 2, 0)])
        property_vectors[(label, 2)] -= c1 * (moments[(label, 0, 1)] + moments[(label, 1, 0)])
        property_vectors[(label, 2)] -= c2 * (moments[(label, 0, 0)])
    t2 = time.time()
    print('LR-SCI-PT compute property vectors... done in', t2 - t1, 's')
    # verify check <N|X|M> = <M|X|N>*
    for label in integrals.keys():
        for N in (0,1,2):
            for M in (0,1,2):
                assert np.allclose(moments[(label, N, M)], moments[(label, M, N)])

    # solve zeroth, first, and second-order linear response equations
    response_vectors = {}
    print('LR-SCI-PT compute response vectors...')
    t1 = time.time()
    dtype = np.complex128 if gamma != 0. else np.float64
    for (label, omega, parity) in perturbations:
        E0w = E0 + parity * (omega + 1j * gamma)
        print(f'{E0w=}')
        
        # zeroth order
        _t1 = time.time()
        rhs = parity*property_vectors[(label, 0)]
        response_vector = np.zeros_like(rhs, dtype=dtype)
        response_vector[:N_internal] = davidson_response(lambda v: matvec(v) - E0w*v, rhs[:N_internal], diagonal[:N_internal]-E0w)
        _t2 = time.time()
        print(f'X0-internal {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[:N_internal] = response_vector[:N_internal] - c0 * np.dot(c0, response_vector[:N_internal])
        response_vector[N_internal:] = rhs[N_internal:] / (diagonal[N_internal:] - E0w)
        response_vectors[(label, omega, parity, 0)] = response_vector
        _t2 = time.time()
        print(f'X0-external {label=}', _t2 - _t1, 's')

        # first order
        _t1 = time.time()
        rhs = parity*property_vectors[(label, 1)].astype(dtype)
        rhs  -= op.Vmatvec_direct(ham, wfn, N_internal, eps2, response_vectors[(label, omega, parity, 0)].real)
        if dtype == np.complex128:
            rhs  -= op.Vmatvec_direct(ham, wfn, N_internal, eps2, response_vectors[(label, omega, parity, 0)].imag)
        rhs[:N_internal] = rhs[:N_internal] - c0 * np.dot(c0, rhs[:N_internal])
        response_vector = np.zeros_like(rhs, dtype=dtype)
        _t2 = time.time()
        print(f'X1-rhs      {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[:N_internal] = davidson_response(lambda v: matvec(v) - E0w*v, rhs[:N_internal], diagonal[:N_internal]-E0w)
        response_vector[:N_internal] = response_vector[:N_internal] - c0 * np.dot(c0, response_vector[:N_internal])
        _t2 = time.time()
        print(f'X1-internal {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[N_internal:] = rhs[N_internal:] / (diagonal[N_internal:] - E0w)
        response_vectors[(label, omega, parity, 1)] = response_vector
        _t2 = time.time()
        print(f'X1-external {label=}', _t2 - _t1, 's')

        # second order
        _t1 = time.time()
        rhs = parity*property_vectors[(label, 2)].astype(dtype)
        rhs -= op.Vmatvec_direct(ham, wfn, N_internal, eps2, response_vectors[(label, omega, parity, 1)].real) 
        if dtype == np.complex128:
            rhs -= op.Vmatvec_direct(ham, wfn, N_internal, eps2, response_vectors[(label, omega, parity, 1)].imag) 
        rhs += E2 * response_vectors[(label, omega, parity, 0)]

        rhs[:N_internal] = rhs[:N_internal] - c0 * np.dot(c0, rhs[:N_internal])
        response_vector = np.zeros_like(rhs, dtype=dtype)
        _t2 = time.time()
        print(f'X2-rhs      {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[:N_internal] = davidson_response(lambda v: matvec(v) - E0w*v, rhs[:N_internal], diagonal[:N_internal]-E0w)
        response_vector[:N_internal] = response_vector[:N_internal] - c0 * np.dot(c0, response_vector[:N_internal])
        _t2 = time.time()
        print(f'X2-internal {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[N_internal:] = rhs[N_internal:] / (diagonal[N_internal:] - E0w)
        response_vectors[(label, omega, parity, 2)] = response_vector
        _t2 = time.time()
        print(f'X2-external {label=}', _t2 - _t1, 's')
    t2 = time.time()
    print('LR-SCI-PT compute response vectors... done in   ', t2 - t1, 's')
    t2 = time.time()

    # assemble perturbative corrections to the response functions
    response_functions = defaultdict(float)
    print('LR-SCI-PT assemble response functions...')
    t1 = time.time()
    for (label, omega, parity) in perturbations:
        for label2 in integrals.keys():
            # zeroth
            response_functions[(label, label2, omega, 0)] += parity * np.dot(response_vectors[(label, omega, parity, 0)], property_vectors[(label2, 0)])
            response_functions[(label, label2, omega, 0, 'internal')] += parity * np.dot(response_vectors[(label, omega, parity, 0)][:N_internal], property_vectors[(label2, 0)][:N_internal])
            response_functions[(label, label2, omega, 0, 'external')] += parity * np.dot(response_vectors[(label, omega, parity, 0)][N_internal:], property_vectors[(label2, 0)][N_internal:])
            # first
            response_functions[(label, label2, omega, 1)] += parity * np.dot(response_vectors[(label, omega, parity, 0)], property_vectors[(label2, 1)])
            response_functions[(label, label2, omega, 1)] += parity * np.dot(response_vectors[(label, omega, parity, 1)], property_vectors[(label2, 0)])
            response_functions[(label, label2, omega, 1, 'internal')] += parity * np.dot(response_vectors[(label, omega, parity, 0)][:N_internal], property_vectors[(label2, 1)][:N_internal])
            response_functions[(label, label2, omega, 1, 'internal')] += parity * np.dot(response_vectors[(label, omega, parity, 1)][:N_internal], property_vectors[(label2, 0)][:N_internal])
            response_functions[(label, label2, omega, 1, 'external')] += parity * np.dot(response_vectors[(label, omega, parity, 0)][N_internal:], property_vectors[(label2, 1)][N_internal:])
            response_functions[(label, label2, omega, 1, 'external')] += parity * np.dot(response_vectors[(label, omega, parity, 1)][N_internal:], property_vectors[(label2, 0)][N_internal:])
            # second
            response_functions[(label, label2, omega, 2)] += parity * np.dot(response_vectors[(label, omega, parity, 0)], property_vectors[(label2, 2)])
            response_functions[(label, label2, omega, 2)] += parity * np.dot(response_vectors[(label, omega, parity, 1)], property_vectors[(label2, 1)])
            response_functions[(label, label2, omega, 2)] += parity * np.dot(response_vectors[(label, omega, parity, 2)], property_vectors[(label2, 0)])
            response_functions[(label, label2, omega, 2, 'internal')] += parity * np.dot(response_vectors[(label, omega, parity, 0)][:N_internal], property_vectors[(label2, 2)][:N_internal])
            response_functions[(label, label2, omega, 2, 'internal')] += parity * np.dot(response_vectors[(label, omega, parity, 1)][:N_internal], property_vectors[(label2, 1)][:N_internal])
            response_functions[(label, label2, omega, 2, 'internal')] += parity * np.dot(response_vectors[(label, omega, parity, 2)][:N_internal], property_vectors[(label2, 0)][:N_internal])
            response_functions[(label, label2, omega, 2, 'external')] += parity * np.dot(response_vectors[(label, omega, parity, 0)][N_internal:], property_vectors[(label2, 2)][N_internal:])
            response_functions[(label, label2, omega, 2, 'external')] += parity * np.dot(response_vectors[(label, omega, parity, 1)][N_internal:], property_vectors[(label2, 1)][N_internal:])
            response_functions[(label, label2, omega, 2, 'external')] += parity * np.dot(response_vectors[(label, omega, parity, 2)][N_internal:], property_vectors[(label2, 0)][N_internal:])
            if omega == 0.0:
                for order in (0,1,2):
                    response_functions[(label, label2, omega, order)] *= 2.0
                    response_functions[(label, label2, omega, order, 'internal')] *= 2.0
                    response_functions[(label, label2, omega, order, 'external')] *= 2.0
    t2 = time.time()
    print('LR-SCI-PT assemble response functions... done in', t2 - t1, 's')
    print_pt2_summary(response_functions)
    return response_functions, property_vectors, response_vectors

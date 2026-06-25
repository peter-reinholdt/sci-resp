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

def resp(ham, wfn, op, e_vecs, integrals, perturbations, frequency=0.0, gamma=0.0, triplet=False, eps_mu=None, eps_resp=None, overwrite=True, state=0, reference_state=None):
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
        overwrite (bool): Allow overwriting of the input wfn and op?
        state (int): index of state to use as the reference state for linear response
        reference_state (np.ndarray): select state based on overlap with this provided reference state
    Returns:
    """
    couple_property = eps_mu is not None
    couple_response = eps_resp is not None

    nroots = state + 1
    
    if not overwrite:
        # avoid overwriting wfn, op
        _wfn = pyci.fullci_wfn(ham.nbasis, wfn.nocc_up, wfn.nocc_dn)
        for det in wfn.to_det_array():
            _wfn.add_det(det)
        wfn = _wfn
        op = pyci.sparse_op(ham, wfn)

    matvec = lambda v: wrap_matvec(op, v)
    hdiag = op.diagonal()
    e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
    while not converged:
        e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
    if reference_state is not None:
        for k in range(len(e_vals)):
            overlap = np.dot(reference_state, e_vecs[:len(reference_state),k])
            print(f'{k=} {overlap=}')
            if np.abs(overlap) > 0.9:
                state = k
        nroots = state + 1
    e_vals += op.ecore
    old_energy = e_vals[state]
    ham_perturbations = {label: pyci.hamiltonian(0., integral, ham.two_mo*0) for (label, integral) in integrals.items()}

    # for each perturbation, add determinants connecting via the one-electron operator, and re-solve
    if couple_property:
        dets_added = 0
        for ham_perturbation in ham_perturbations.values():
            dets_added += pyci.add_hci(ham_perturbation, wfn, e_vecs[:, state], eps=eps_mu)
            num_determinants = len(wfn)
        e_vecs = zeropad(e_vecs, num_determinants)
        op.update(ham, wfn)
        matvec = lambda v: wrap_matvec(op, v)
        hdiag = op.diagonal()
        e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
        while not converged:
            e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
        if reference_state is not None:
            for k in range(len(e_vals)):
                overlap = np.dot(reference_state, e_vecs[:len(reference_state),k])
                print(f'{k=} {overlap=}')
                if np.abs(overlap) > 0.9:
                    state = k
            nroots = state + 1
        e_vals += op.ecore
        delta_e = old_energy - e_vals[state]
        old_energy = e_vals[state]
        for root in range(nroots):
            istarget = ' * ' if root == state else '   '
            ΔE_au = e_vals[root] - e_vals[0]
            ΔE_eV = (e_vals[root] - e_vals[0]) * 27.211396
            print(f'{root=}{istarget} {e_vals[root]=: 16.12f} {ΔE_au=: 12.8f} {ΔE_eV=: 12.8f}')
        print(f'couple_property {eps_mu=} {e_vals[state]=} {delta_e=} {dets_added=} {num_determinants=}')

    # form property vectors
    property_vectors = {}
    for (label, omega, parity) in perturbations:
        if label in property_vectors:
            continue
        ham_perturbation = ham_perturbations[label]
        property_vector = ham_perturbation.one_electron_direct(wfn, e_vecs[:,state], triplet=triplet)
        property_vector -= e_vecs[:,state] * np.dot(e_vecs[:,state], property_vector)
        property_vectors[label] = property_vector

    # solve response equations
    E0 = np.dot(e_vecs[:,state], matvec(e_vecs[:,state]))
    response_vectors = {}
    for (label, omega, parity) in perturbations:
        E0w = E0 + parity * (omega + 1j * gamma)
        property_vector = property_vectors[label]
        response_vector = davidson_response(lambda v: matvec(v)-E0w*v, property_vector, hdiag-E0w, verbose=False)
        response_vector -= e_vecs[:, state] * np.dot(e_vecs[:, state].ravel(), response_vector.ravel())
        response_vectors[(label, omega, parity)] = response_vector

    # add determinants coupling through response vector (and solve response equations again)
    if couple_response:
        while True:
            # add determinants
            dets_added = 0
            screen_vector = np.zeros(len(wfn))
            for (label, omega, parity), response_vector in response_vectors.items():
                screen_vector = np.max([np.abs(response_vector), screen_vector], axis=0)
            dets_added = pyci.add_hci(ham, wfn, screen_vector, eps=eps_resp)
            del screen_vector
            num_determinants = len(wfn)
            e_vecs = zeropad(e_vecs, num_determinants)
            op.update(ham, wfn)
            matvec = lambda v: wrap_matvec(op, v)
            hdiag = op.diagonal()
            e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
            while not converged:
                e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=nroots, c0=e_vecs, verbose=True, reference_state=reference_state)
            if reference_state is not None:
                for k in range(len(e_vals)):
                    overlap = np.dot(reference_state, e_vecs[:len(reference_state),k])
                    print(f'{k=} {overlap=}')
                    if np.abs(overlap) > 0.9:
                        state = k
                nroots = state + 1
            e_vals += op.ecore
            delta_e = old_energy - e_vals[state]
            old_energy = e_vals[state]
            # form property vectors
            property_vectors = {}
            for (label, omega, parity) in perturbations:
                if label in property_vectors:
                    continue
                ham_perturbation = ham_perturbations[label]
                property_vector = ham_perturbation.one_electron_direct(wfn, e_vecs[:,state], triplet=triplet)
                property_vector -= e_vecs[:,state] * np.dot(e_vecs[:,state], property_vector)
                property_vectors[label] = property_vector
            # solve response equations
            E0 = np.dot(e_vecs[:,state], matvec(e_vecs[:,state]))
            for (label, omega, parity) in perturbations:
                E0w = E0 + parity * (omega + 1j * gamma)
                property_vector = property_vectors[label]
                guess = zeropad(response_vectors[(label, omega, parity)], num_determinants)
                response_vector = davidson_response(lambda v: matvec(v)-E0w*v, property_vector, hdiag-E0w, verbose=False, guess=guess)
                response_vector -= e_vecs[:, state] * np.dot(e_vecs[:, state].ravel(), response_vector.ravel())
                response_vectors[(label, omega, parity)] = response_vector
            # exit when almost no new determinants are added
            for root in range(nroots):
                istarget = ' * ' if root == state else '   '
                ΔE_au = e_vals[root] - e_vals[0]
                ΔE_eV = (e_vals[root] - e_vals[0]) * 27.211396
                print(f'{root=}{istarget} {e_vals[root]=: 16.12f} {ΔE_au=: 12.8f} {ΔE_eV=: 12.8f}')
            if dets_added / num_determinants < 0.01:
                print(f'finished_couple_response {eps_resp=} {e_vals[state]=} {delta_e=} {dets_added=} {num_determinants=}')
                break
            else:
                print(f'couple_response {eps_resp=} {e_vals[state]=} {delta_e=} {dets_added=} {num_determinants=}')
    response_functions = defaultdict(float)
    for (label, omega, parity) in perturbations:
        for label2 in integrals.keys():
            response_functions[(label, label2, omega)] += np.dot(response_vectors[(label, omega, parity)], property_vectors[label2])
    # for static we have used only the positive parity
    for (label, omega, parity) in perturbations:
            if (omega == 0.0) and (gamma == 0.0):
                for label2 in integrals.keys():
                    response_functions[(label, label2, omega)] *= 2.0
    return wfn, op, e_vecs, response_vectors, property_vectors, response_functions


def print_var_summary(response_functions):
    labels = [key for key in response_functions.keys()]
    for (label1, label2, omega) in labels:
        value = response_functions[(label1, label2, omega)]
        ω = omega
        if value.imag != 0:
            imag = value.imag
            value = value.real
            print(f'Re<<{label1}; {label2}>>({ω=:}) {value=:16.9f}')
            value = imag
            print(f'Im<<{label1}; {label2}>>({ω=:}) {value=:16.9f}')
        else:
            value = value.real
            print(f'<<{label1}; {label2}>>({ω=:}) {value=:16.9f}')

def print_pt2_summary(response_functions):
    labels = [key for key in response_functions.keys() if key[-1] == 2]
    for label in labels:
        label1, label2, omega, _ = label
        Δ_0 = response_functions[(label1, label2, omega, 0)]
        Δ_1 = response_functions[(label1, label2, omega, 1)]
        Δ_2 = response_functions[(label1, label2, omega, 2)]
        value = Δ_0 + Δ_1 + Δ_2
        ω = omega
        if value.imag != 0:
            # complex
            imag = value.imag, Δ_0.imag, Δ_1.imag, Δ_2.imag
            value = value.real
            Δ_0 = Δ_0.real
            Δ_1 = Δ_1.real
            Δ_2 = Δ_2.real
            print(f'Re<<{label1}; {label2}>>({ω=:}) {value=:16.9f} {Δ_0=:16.9f} {Δ_1=:16.9f} {Δ_2=:16.9f}')
            value, Δ_0, Δ_1, Δ_2 = imag
            print(f'Im<<{label1}; {label2}>>({ω=:}) {value=:16.9f} {Δ_0=:16.9f} {Δ_1=:16.9f} {Δ_2=:16.9f}')
        else:
            value = value.real
            print(f'<<{label1}; {label2}>>({ω=:}) {value=:16.9f} {Δ_0=:16.9f} {Δ_1=:16.9f} {Δ_2=:16.9f}')


def resp_pt2(ham, wfn, op, e_vecs, integrals, perturbations, eps2, eps2mult, frequency=0.0, gamma=0.0, triplet=False, eps_mu=None, eps_resp=None, overwrite=True, state=0, reference_state=None):
    # solve the internal/variational LR problem
    # (we run GS, GS+V, GS+X, or GS+V+X)
    #
    if not overwrite:
        # avoid overwriting wfn, op
        _wfn = pyci.fullci_wfn(ham.nbasis, wfn.nocc_up, wfn.nocc_dn)
        for det in wfn.to_det_array():
            _wfn.add_det(det)
        wfn = _wfn
        op = pyci.sparse_op(ham, wfn)

    print('LR-SCI-PT solving variational equations...')
    t1 = time.time()
    wfn, op, e_vecs, response_vectors, property_vectors, response_functions = resp(ham, wfn, op, e_vecs, integrals, perturbations, frequency=frequency, gamma=gamma, triplet=triplet, eps_mu=eps_mu, eps_resp=eps_resp, state=state, reference_state=reference_state)
    t2 = time.time()
    print('LR-SCI-PT solving variational equations... done in', t2 - t1, 's')
    # check if reference state was re-ordered
    if reference_state is not None:
        for k in range(e_vecs.shape[1]):
            overlap = np.dot(reference_state, e_vecs[:len(reference_state),k])
            print(f'{k=} {overlap=}')
            if np.abs(overlap) > 0.9:
                state = k
    c0 = e_vecs[:, state].ravel()
    E0 = np.dot(c0, op.matvec(c0))

    N_internal = op.shape[0]
    def matvec(v, c0=c0):
        v = v - c0*np.dot(c0, v)
        Hv = op.matvec(v)
        Hv = Hv - c0 * np.dot(c0, Hv)
        return Hv

    # select perturbative space for c1, c2
    print(f'LR-SCI-PT c0-det-add...')
    t1 = time.time()
    added = pyci.add_hci(ham, wfn, c0, eps=eps2)
    t2 = time.time()
    print(f'LR-SCI-PT c0-det-add...', t2 - t1, 's')

    # get c1 and c2
    # update diagonal
    t1 = time.time()
    op.update_diagonal(ham, wfn)
    diagonal = op.diagonal()
    N_total = len(diagonal)
    N_external = N_total - N_internal
    t2 = time.time()
    print('LR-SCI-PT diagonal... done in: ', t2 - t1, 's')

    # compute first-order wave function coefficients (c1)
    print('LR-SCI-PT V@c0...')
    t1 = time.time()
    Vc0 = op.Vmatvec_direct(ham, wfn, N_internal, eps2mult, c0)
    print('LR-SCI-PT forming c1...')
    c1 = Vc0 / (E0 - diagonal)
    E2 = np.dot(c1, Vc0)
    print(f'PT2-corrected energy is {E0 + E2=}')
    del Vc0
    t2 = time.time()
    print('LR-SCI-PT V@c0... done in:     ', t2 - t1, 's')

    # compute second-order wave function coefficients (c2)
    print('LR-SCI-PT V@c1...')
    t1 = time.time()
    rhs = -op.Vmatvec_direct(ham, wfn, N_internal, eps2mult, c1)
    t2 = time.time()
    print('LR-SCI-PT V@c1... done in:     ', t2 - t1, 's')
    print('LR-SCI-PT forming c2...')
    t1 = time.time()
    c2 = np.zeros_like(c1)
    rhs[:N_internal] += E2 * c0
    c2[N_internal:] = rhs[N_internal:] / (diagonal[N_internal:] - E0)
    t2 = time.time()
    c2[:N_internal] = davidson_response(lambda v: matvec(v) - E0*v, rhs[:N_internal], diagonal[:N_internal]-E0, verbose=True)
    c2[:N_internal] -= c0 * np.dot(c0, c2[:N_internal])
    t2 = time.time()
    print('LR-SCI-PT forming c2... done in', t2 - t1, 's')


    # add from property operator
    print(f'LR-SCI-PT det-add...')
    t1 = time.time()
    dets_added = 0
    ham_perturbations = {label: pyci.hamiltonian(0., integral, ham.two_mo*0) for (label, integral) in integrals.items()}
    for (label, ham_perturbation) in ham_perturbations.items():
        added = pyci.add_hci(ham_perturbation, wfn, c0, eps=eps2)
        print('Added', added, 'from operator', label)
        dets_added += added
    # add from Hamiltonian / response vectors
    screen_vector = np.zeros_like(c0)
    for (label, omega, parity), response_vector in response_vectors.items():
        screen_vector = np.max([np.abs(response_vector), screen_vector], axis=0)
    added = pyci.add_hci(ham, wfn, screen_vector, eps=eps2)
    del screen_vector
    print('Added', added, 'from response vectors')
    t2 = time.time()
    print('LR-SCI-PT det-add... done in:  ', t2 - t1, 's')

    t1 = time.time()
    op.update_diagonal(ham, wfn)
    diagonal = op.diagonal()
    N_total = len(diagonal)
    N_external = N_total - N_internal
    t2 = time.time()
    print('LR-SCI-PT diagonal... done in: ', t2 - t1, 's')

    print(f'LR-SCI-PT epsilons  : {eps2=} {eps2mult=} {eps_mu=} {eps_resp=}')
    print(f'LR-SCI-PT dimensions: {N_internal=} {N_external=} {N_total=}')



    # compute zeroth, first, and second-order one-electron operator products
    # and form zeroth, first, and second-order property vectors
    property_vectors = {}
    moments = {}
    t1 = time.time()
    print('LR-SCI-PT compute property vectors...')
    for (label, omega, parity) in perturbations:
        if (label, 0) in property_vectors:
            continue
        ham_perturbation = ham_perturbations[label]
        # zeroth order
        print(f'{label=}, order=0')
        product = ham_perturbation.one_electron_direct(wfn, c0, triplet=triplet)
        moments[(label, 0, 0)] = np.dot(c0, product[:len(c0)])
        moments[(label, 1, 0)] = np.dot(c1, product[:len(c1)])
        moments[(label, 2, 0)] = np.dot(c2, product[:len(c2)])
        property_vectors[(label, 0)] = product
        property_vectors[(label, 0)][:len(c0)] -= c0 * moments[(label, 0, 0)]

        # first order
        print(f'{label=}, order=1')
        product = ham_perturbation.one_electron_direct(wfn, c1, triplet=triplet)
        moments[(label, 0, 1)] = np.dot(c0, product[:len(c0)])
        moments[(label, 1, 1)] = np.dot(c1, product[:len(c1)])
        moments[(label, 2, 1)] = np.dot(c2, product[:len(c2)])
        property_vectors[(label, 1)] = product
        property_vectors[(label, 1)][:len(c0)] -= c0 * (moments[(label, 0, 1)] + moments[(label, 1, 0)])
        property_vectors[(label, 1)][:len(c1)] -= c1 * (moments[(label, 0, 0)])

        # second order
        print(f'{label=}, order=2')
        product = ham_perturbation.one_electron_direct(wfn, c2, triplet=triplet)
        moments[(label, 0, 2)] = np.dot(c0, product[:len(c0)])
        moments[(label, 1, 2)] = np.dot(c1, product[:len(c1)])
        moments[(label, 2, 2)] = np.dot(c2, product[:len(c2)])
        property_vectors[(label, 2)] = product
        property_vectors[(label, 2)][:len(c0)] -= c0 * (moments[(label, 0, 2)] + moments[(label, 1, 1)] + moments[(label, 2, 0)])
        property_vectors[(label, 2)][:len(c1)] -= c1 * (moments[(label, 0, 1)] + moments[(label, 1, 0)])
        property_vectors[(label, 2)][:len(c2)] -= c2 * (moments[(label, 0, 0)])
    # c1, c2 no longer needed
    del c1
    del c2
    t2 = time.time()
    print('LR-SCI-PT compute property vectors... done in', t2 - t1, 's')
    # verify check <N|X|M> = <M|X|N>*
    for label in integrals.keys():
        for N in (0,1,2):
            for M in (0,1,2):
                if not (np.allclose(moments[(label, N, M)], moments[(label, M, N)]) or np.allclose(moments[(label, N, M)], -moments[(label, M, N)])):
                    print(f'WARNING: transition moments did not match for {(label, M, N)} and {(label, M, N)}:', moments[(label, N, M)], moments[(label, M, N)])



    # solve zeroth, first, and second-order linear response equations
    # and assemble final response functions
    print('LR-SCI-PT compute response vectors...')
    t1 = time.time()
    dtype = np.complex128 if gamma != 0. else np.float64
    response_functions = defaultdict(float)
    # do zeroth order response
    for (label, omega, parity) in perturbations:
        _t1 = time.time()
        E0w = E0 + parity * (omega + 1j * gamma)
        rhs = property_vectors[(label, 0)].astype(dtype)
        response_vector = np.zeros_like(rhs, dtype=dtype)
        response_vector[:N_internal] = davidson_response(lambda v: matvec(v) - E0w*v, rhs[:N_internal],
                diagonal[:N_internal]-E0w, guess=response_vectors[(label, omega, parity)])
        _t2 = time.time()
        print(f'X0-internal {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[:N_internal] -= c0 * np.dot(c0, response_vector[:N_internal])
        response_vector[N_internal:] = rhs[N_internal:] / (diagonal[N_internal:] - E0w)
        _t2 = time.time()
        print(f'X0-external {label=}', _t2 - _t1, 's')
        response_vectors[(label, omega, parity)] = response_vector
    # assemble contributions that require zeroth-order response vector
    for (label, omega, parity) in perturbations:
        _t1 = time.time()
        for label2 in integrals.keys():
            # zeroth-order (0,0) <XA0|B0>
            response_functions[(label, label2, omega, 0)] += np.dot(response_vectors[(label, omega, parity)], property_vectors[(label2, 0)])
            # first-order (0,1)  <XA0|B1>
            response_functions[(label, label2, omega, 1)] += np.dot(response_vectors[(label, omega, parity)], property_vectors[(label2, 1)])
            # second-order (0,2) <XA0|B2>
            response_functions[(label, label2, omega, 2)] += np.dot(response_vectors[(label, omega, parity)], property_vectors[(label2, 2)])
            # second-order (2,0) <XB2|A0> = <XB0|A2> - <XB0|V|XA1> + E2 <XA0|XB0>
            # we add <XB0|A2> and E2 <XA0|XB0> here, <XB0|V|XA1> done when we have XA1 and V@XB0
            response_functions[(label, label2, omega, 2)] += np.dot(response_vectors[(label2, omega, parity)], property_vectors[(label, 2)])
            response_functions[(label, label2, omega, 2)] += E2 * np.dot(response_vectors[(label, omega, parity)], response_vectors[(label2, omega, parity)])
        _t2 = time.time()
        print(f'X0-dots {label=}', _t2 - _t1, 's')
    # A2 no longer needed
    for label in integrals.keys():
        del property_vectors[(label, 2)]

    # compute and store VX0
    VX0_vectors = response_vectors
    for (label, omega, parity) in perturbations:
        _t1 = time.time()
        response_vector = response_vectors[(label, omega, parity)]
        VX0 = np.zeros_like(response_vector, dtype=dtype)
        VX0 += op.Vmatvec_direct(ham, wfn, N_internal, eps2mult, response_vector.real)
        if dtype == np.complex128:
            VX0 += 1j * op.Vmatvec_direct(ham, wfn, N_internal, eps2mult, response_vector.imag)
        VX0_vectors[(label, omega, parity)] = VX0
        _t2 = time.time()
        print(f'V@X0 {label=}', _t2 - _t1, 's')
    # do first-order response
    for (label, omega, parity) in perturbations:
        E0w = E0 + parity * (omega + 1j * gamma)
        _t1 = time.time()
        response_vector = np.zeros_like(rhs, dtype=dtype)
        rhs[:] = property_vectors[(label, 1)] - VX0_vectors[(label, omega, parity)]
        rhs[:N_internal] -= c0 * np.dot(c0, rhs[:N_internal])
        _t2 = time.time()
        print(f'X1-rhs      {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[:N_internal] = davidson_response(lambda v: matvec(v) - E0w*v, rhs[:N_internal], diagonal[:N_internal]-E0w)
        response_vector[:N_internal] -= c0 * np.dot(c0, response_vector[:N_internal])
        _t2 = time.time()
        print(f'X1-internal {label=}', _t2 - _t1, 's')
        _t1 = time.time()
        response_vector[N_internal:] = rhs[N_internal:] / (diagonal[N_internal:] - E0w)
        _t2 = time.time()
        print(f'X1-external {label=}', _t2 - _t1, 's')

        _t1 = time.time()
        # assemble contributions that require zeroth-order response vector
        for label2 in integrals.keys():
            # first-order (1,0)  <XA1|B0>
            response_functions[(label, label2, omega, 1)] += np.dot(response_vector, property_vectors[(label2, 0)])
            # second-order (1,1) <XA1|B1>
            response_functions[(label, label2, omega, 2)] += np.dot(response_vector, property_vectors[(label2, 1)])
            # second-order (2,0) <XB2|A0> = <XB0|A2> - <XB0|V|XA1> + E2 <XA0|XB0>
            # we add the -<XB0|V|XA1> contribution here, <XB0|A2> + E2 <XA0|XB0> was added with the zeroth-order vectors
            response_functions[(label, label2, omega, 2)] -= np.dot(response_vector, VX0_vectors[(label2, omega, parity)])
        _t2 = time.time()
        print(f'X1-dots {label=}', _t2 - _t1, 's')
    t2 = time.time()
    print('LR-SCI-PT compute response vectors... done in   ', t2 - t1, 's')

    # for static we have used only the positive parity
    for (label, omega, parity) in perturbations:
            if (omega == 0.0) and (gamma == 0.0):
                for label2 in integrals.keys():
                    for order in (0,1,2):
                        response_functions[(label, label2, omega, order)] *= 2.0
    # also store the total value
    for (label, omega, parity) in perturbations:
        for label2 in integrals.keys():
            for order in (0,1,2):
                response_functions[(label, label2, omega)] += response_functions[(label, label2, omega, order)]
    print_pt2_summary(response_functions)
    return response_functions

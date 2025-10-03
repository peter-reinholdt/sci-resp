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
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.data import nist
import scipy

parser = argparse.ArgumentParser()
parser.add_argument('--couple-property', type=bool, default=True)
parser.add_argument('--couple-response', type=bool, default=True)
parser.add_argument('--eps', type=float, default=1e-3)
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
basis = '6-31G-J'
nuc_pair = [(1,2), (1,0)]

m = pyscf.M(atom=xyz, basis=basis, symmetry=True)

ncore = 0
ncas = m.nao
nelcas = sum(m.nelec)
nelec = m.nelec

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
            #mu_op.update(ham_mu, wfn)
            e_vecs = np.concatenate((e_vecs, np.zeros((dets_added, e_vecs.shape[1]))), axis=0)
            matvec = lambda v: wrap_matvec(op, v)
            hdiag = op.diagonal()
            e_vals, e_vecs = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
            e_vals += op.ecore
            delta_e = old_energy - np.min(e_vals)
            old_energy = np.min(e_vals)
            num_determinants = e_vecs.shape[0]
            #zPsi = mu_op.matvec(e_vecs)
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

def dso_integral(mol, orig1, orig2):
    '''Integral of vec{r}vec{r}/(|r-orig1|^3 |r-orig2|^3)
    Ref. JCP, 73, 5718'''
    NUMINT_GRIDS = 30
    from pyscf import gto
    t, w = np.polynomial.legendre.leggauss(NUMINT_GRIDS)
    a = (1+t)/(1-t) * .8
    w *= 2/(1-t)**2 * .8
    fakemol = gto.Mole()
    fakemol._atm = np.asarray([[0, 0, 0, 0, 0, 0]], dtype=np.int32)
    fakemol._bas = np.asarray([[0, 1, NUMINT_GRIDS, 1, 0, 3, 3+NUMINT_GRIDS, 0]],
                                 dtype=np.int32)
    p_cart2sph_factor = 0.488602511902919921
    fakemol._env = np.hstack((orig2, a**2, a**2*w*4/np.pi**.5/p_cart2sph_factor))
    fakemol._built = True

    pmol = mol + fakemol
    pmol.set_rinv_origin(orig1)
    # <nabla i, j | k>  k is a fictitious basis for numerical integraion
    mat1 = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, pmol.nbas))
    # <i, j | nabla k>
    mat  = pmol.intor(mol._add_suffix('int3c1e_iprinv'), comp=3,
                      shls_slice=(mol.nbas, pmol.nbas, 0, mol.nbas, 0, mol.nbas))
    mat += mat1.transpose(0,3,1,2) + mat1.transpose(0,3,2,1)
    return mat

def _atom_gyro_list(mol):
    gyro = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in mol.nucprop:
            prop = mol.nucprop[symb]
            mass = prop.get('mass', None)
            gyro.append(get_nuc_g_factor(symb, mass))
        else:
            # Get default isotope
            gyro.append(get_nuc_g_factor(symb))
    return np.array(gyro)

def convert_unit(e11):
    # unit conversions
    e11 = e11*nist.ALPHA**4
    nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2Hz = nist.HARTREE2J / nist.PLANCK
    unit = au2Hz * nuc_magneton ** 2
    iso_ssc = unit * np.einsum('kii->k', e11) / 3
    natm = m.natm
    ktensor = np.zeros((natm,natm))
    for k, (i, j) in enumerate(nuc_pair):
        ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
    gyro = _atom_gyro_list(m)
    jtensor = np.einsum('ij,i,j->ij', ktensor, gyro, gyro)
    return jtensor

def response_dot(response_wfn, response_vector, response_e_vecs, h1ao, triplet=None):
    #             ~
    # evaluate <X|B>; form property vector B from the determinant set used in response vector X
    property_operator = triplet_operator_state if triplet else singlet_operator_state
    mu =  one_electron_operator_for_cas(cas, h1ao)
    property_vector = property_operator(response_wfn.to_det_array(), mu, response_e_vecs.ravel())
    property_vector -= response_e_vecs.ravel() * np.dot(property_vector, response_e_vecs.ravel())
    return np.dot(response_vector, property_vector)

# SSCC - DSO (expectation value)
e11_dso = np.zeros((len(nuc_pair), 3, 3))
d1, d2 = pyci.compute_rdms(wfn, e_vecs[0])

rdm1 = d1[0] + d1[1]
rdm1 = _make_rdm1_on_mo(rdm1, ncore, ncas, m.nao)
rdm1_ao = mf.mo_coeff @ rdm1 @ mf.mo_coeff.T
for k, (i,j) in enumerate(nuc_pair):
    dso_ao = dso_integral(m, m.atom_coord(i), m.atom_coord(j))
    a11 = -np.einsum('xymn,mn->xy', dso_ao, rdm1_ao)
    a11 = a11 - a11.trace() * np.eye(3)
    e11_dso[k] = a11

# SSCC - PSO (response)
e11_pso = np.zeros((len(nuc_pair), 3, 3))
response_vectors = []
response_e_vecs = []
response_wfns = []
h1aos = []
for ia in range(m.natm):
    m.set_rinv_origin(m.atom_coord(ia))
    h1ao = m.intor_asymmetric('int1e_prinvxp', 3)
    for operator in h1ao.reshape(3, m.nao, m.nao):
        # we can skip response on this nucleus if it is not needed
        # only the i'th perturbation in (i,j) nuc_pair is required
        # we still need to save the operator for the property gradient
        if not any([ia in pair[:1] for pair in nuc_pair]):
            response_wfns.append(None)
            response_e_vecs.append(None)
            response_vectors.append(None)
            h1aos.append(operator)
            continue
        wfn_rsp, response_e_vec, response_vector, property_vector = resp(ham, nelec, wfn, e_vecs, operator, omega=0.0, gamma=0.0, couple_property=args.couple_property, couple_response=args.couple_response, triplet=False)
        response_wfns.append(wfn_rsp)
        response_e_vecs.append(response_e_vec)
        response_vectors.append(response_vector)
        h1aos.append(operator)

for k, (i,j) in enumerate(nuc_pair):
    # 'xi,yi->xy'
    response_vectors_i = response_vectors[i*3:(i+1)*3] 
    response_wfns_i = response_wfns[i*3:(i+1)*3]
    response_e_vecs_i = response_e_vecs[i*3:(i+1)*3]
    h1aos_j = h1aos[j*3:(j+1)*3]
    for x in range(3):
        for y in range(3):
            e11_pso[k,x,y] += -2*response_dot(response_wfns_i[x], response_vectors_i[x], response_e_vecs_i[x], h1aos_j[y], triplet=False)

# SSCC - SD (response)
e11_sd = np.zeros((len(nuc_pair), 3, 3))
response_vectors = []
response_e_vecs = []
response_wfns = []
h1aos = []

for ia in range(m.natm):
    m.set_rinv_origin(m.atom_coord(ia))
    a01p = nist.G_ELECTRON * 0.25 * m.intor('int1e_sa01sp', 12).reshape(3,4,m.nao,m.nao)
    h1ao = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2))
    
    # remove FC from FC+SD integral
    coords = m.atom_coord(ia).reshape(1, 3)
    ao = m.eval_gto('GTOval', coords)
    fc = 8*np.pi/3 * np.einsum('ip,iq->pq', ao, ao) * (nist.G_ELECTRON/2) / 2
    h1ao -= np.einsum('xy,mn->xymn', np.eye(3), fc)
    for i, operator in enumerate(h1ao.reshape(9, m.nao, m.nao)):
        if not any([ia in pair[:1] for pair in nuc_pair]):
            response_wfns.append(None)
            response_e_vecs.append(None)
            response_vectors.append(None)
            h1aos.append(operator)
            continue
        wfn_rsp, response_e_vec, response_vector, property_vector = resp(ham, nelec, wfn, e_vecs, operator, omega=0.0, gamma=0.0, couple_property=args.couple_property, couple_response=args.couple_response, triplet=True)
        response_wfns.append(wfn_rsp)
        response_e_vecs.append(response_e_vec)
        response_vectors.append(response_vector)
        h1aos.append(operator)

for k, (i,j) in enumerate(nuc_pair):
    # 'xwi,ywi->xy'
    response_vectors_i = response_vectors[i*9:(i+1)*9] 
    response_wfns_i = response_wfns[i*9:(i+1)*9]
    response_e_vecs_i = response_e_vecs[i*9:(i+1)*9]
    h1aos_j = h1aos[j*9:(j+1)*9]
    for x in range(3):
        for y in range(3):
            for w in range(3):
                xw = 3*x+w
                yw = 3*y+w
                e11_sd[k,x,y] += -2*response_dot(response_wfns_i[xw], response_vectors_i[xw], response_e_vecs_i[xw], h1aos_j[yw], triplet=True)

# FC
e11_fc = np.zeros((len(nuc_pair), 3, 3))
response_vectors = []
response_e_vecs = []
response_wfns = []
h1aos = []
for ia in range(m.natm):
    coords = m.atom_coord(ia).reshape(1, 3)
    ao = m.eval_gto('GTOval', coords)
    h1ao = 8*np.pi/3 * np.einsum('ip,iq->pq', ao, ao) * (nist.G_ELECTRON/2) / 2
    operator = h1ao
    if not any([ia in pair[:1] for pair in nuc_pair]):
        response_wfns.append(None)
        response_e_vecs.append(None)
        response_vectors.append(None)
        h1aos.append(operator)
        continue
    wfn_rsp, response_e_vec, response_vector, property_vector = resp(ham, nelec, wfn, e_vecs, operator, omega=0.0, gamma=0.0, couple_property=args.couple_property, couple_response=args.couple_response, triplet=True)
    response_wfns.append(wfn_rsp)
    response_e_vecs.append(response_e_vec)
    response_vectors.append(response_vector)
    h1aos.append(operator)

for k, (i,j) in enumerate(nuc_pair):
    # 'xwi,ywi->xy'
    response_vectors_i = response_vectors[i]
    response_wfns_i = response_wfns[i]
    response_e_vecs_i = response_e_vecs[i]
    h1aos_j = h1aos[j]
    val = -2*response_dot(response_wfns_i, response_vectors_i, response_e_vecs_i, h1aos_j, triplet=True)
    e11_fc[k,0,0] = e11_fc[k,1,1] = e11_fc[k,2,2] = val


print('SSCC (in Hz):')
j_tensor_fc = convert_unit(e11_fc)
j_tensor_sd = convert_unit(e11_sd)
j_tensor_pso = convert_unit(e11_pso)
j_tensor_dso = convert_unit(e11_dso)
j_tensor_total = j_tensor_fc + j_tensor_sd + j_tensor_pso + j_tensor_dso
for (i,j) in nuc_pair:
    print(f'{i} {j}:  DSO={j_tensor_dso[i,j]:.6f}, PSO={j_tensor_pso[i,j]:.6f}, FC={j_tensor_fc[i,j]:.6f}, SD={j_tensor_sd[i,j]:.6f}, Total={j_tensor_total[i,j]:.6f}')

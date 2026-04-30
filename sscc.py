#!/usr/bin/env python

import sys
import time
import argparse
import pyscf
import numpy as np
import pyci
from solvers import solve_ci
from response import resp, wrap_matvec, _make_rdm1_on_mo, one_electron_ao2mo
from sscc_utils import dso_integral, pso_integrals, sd_integrals, fc_integrals, atom_gyro_list, convert_unit

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', type=str, required=True)
parser.add_argument('--basis', type=str, required=True)
parser.add_argument('--ncore', type=int, default=0)
parser.add_argument('--couple-property', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--couple-response', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eps', type=float, default=1e-3)
args = parser.parse_args()

xyz = args.xyz
basis = args.basis

mol = pyscf.M(atom=xyz, basis=basis, symmetry=True)
#nuc_pair = [(i,j) for i in range(mol.natm) for j in range(i)]
active_atoms = [1,2]
nuc_pair = [(i,j) for i in active_atoms for j in active_atoms if i<j]

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
cas = pyscf.mcscf.CASCI(mf, ncas, nelcas)

h1, ecore = cas.h1e_for_cas()
eri = pyscf.ao2mo.full(mol, mf.mo_coeff[:, ncore:ncore+ncas], aosym='1').reshape(ncas, ncas, ncas, ncas)
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
    e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
    while not converged:
        e_vals, e_vecs, converged = solve_ci(matvec, hdiag, roots=1, c0=e_vecs, verbose=True)
    e_vals += op.ecore
    delta_e = old_energy - np.min(e_vals)
    old_energy = np.min(e_vals)
    niter += 1
    num_determinants = e_vecs.shape[0]
    print(f'{niter=} {eps=} {e_vals[0]=} {delta_e=} {dets_added=} {num_determinants=}')

# SSCC - DSO (expectation value)
e11_dso = np.zeros((len(nuc_pair), 3, 3))
e11_pso = np.zeros((len(nuc_pair), 3, 3))
e11_sd = np.zeros((len(nuc_pair), 3, 3))
e11_fc = np.zeros((len(nuc_pair), 3, 3))
d1 = pyci.compute_rdm1(wfn, e_vecs[0])

rdm1 = d1[0] + d1[1]
rdm1 = _make_rdm1_on_mo(rdm1, ncore, ncas, mol.nao)
rdm1_ao = mf.mo_coeff @ rdm1 @ mf.mo_coeff.T
for k, (i,j) in enumerate(nuc_pair):
    dso_ao = dso_integral(mol, mol.atom_coord(i), mol.atom_coord(j))
    a11 = -np.einsum('xymn,mn->xy', dso_ao, rdm1_ao)
    a11 = a11 - a11.trace() * np.eye(3)
    e11_dso[k] = a11


# SSCC - PSO (singlet response)
pso_integrals_ao = pso_integrals(mol, active_atoms)
pso_integrals_mo = [one_electron_ao2mo(cas, integral) for integral in pso_integrals_ao]
labels = [f'PSO_{ia}_{cart}' for ia in active_atoms for cart in 'xyz']
integrals  = {label: integral for (label, integral) in zip(labels, pso_integrals_mo)}
perturbations = [(label, frequency, parity) for label in labels 
                                            for frequency in [0.0] 
                                            for parity in [0]]
rsp_wfn, rsp_op, rsp_e_vecs, response_vectors, property_vectors, response_functions = resp(ham, wfn, op, e_vecs, integrals, perturbations, eps_mu=eps, eps_resp=eps, triplet=False, overwrite=False)
for k, (i,j) in enumerate(nuc_pair):
    for ix, x in enumerate('xyz'):
        for iy, y in enumerate('xyz'):
            label1 = f'PSO_{i}_{x}'
            label2 = f'PSO_{j}_{y}'
            e11_pso[k, ix, iy] = -response_functions[(label1, label2, 0.0)]

# SSCC - SD (triplet response)
sd_integrals_ao = sd_integrals(mol, active_atoms)
sd_integrals_mo = [one_electron_ao2mo(cas, integral) for integral in sd_integrals_ao]
labels = [f'SD_{ia}_{cart1}{cart2}' for ia in active_atoms for cart1 in 'xyz' for cart2 in 'xyz']
integrals  = {label: integral for (label, integral) in zip(labels, sd_integrals_mo) if label[-2] <= label[-1]}
perturbations = [(label, frequency, parity) for label in labels 
                                            for frequency in [0.0] 
                                            for parity in [0] if label[-2] <= label[-1]]
rsp_wfn, rsp_op, rsp_e_vecs, response_vectors, property_vectors, response_functions = resp(ham, wfn, op, e_vecs, integrals, perturbations, eps_mu=eps, eps_resp=eps, triplet=True, overwrite=False)
for k, (i,j) in enumerate(nuc_pair):
    for ix, x in enumerate('xyz'):
        for iy, y in enumerate('xyz'):
            for iw, w in enumerate('xyz'):
                xw = ''.join(sorted(f'{x}{w}'))
                yw = ''.join(sorted(f'{y}{w}'))
                label1 = f'SD_{i}_{xw}'
                label2 = f'SD_{j}_{yw}'
                e11_sd[k, ix, iy] += -response_functions[(label1, label2, 0.0)]

# SSCC - FC (triplet response)
fc_integrals_ao = fc_integrals(mol, active_atoms)
fc_integrals_mo = [one_electron_ao2mo(cas, integral) for integral in fc_integrals_ao]
labels = [f'FC_{ia}' for ia in active_atoms]
integrals  = {label: integral for (label, integral) in zip(labels, fc_integrals_mo)}
perturbations = [(label, frequency, parity) for label in labels 
                                            for frequency in [0.0] 
                                            for parity in [0]]
rsp_wfn, rsp_op, rsp_e_vecs, response_vectors, property_vectors, response_functions = resp(ham, wfn, op, e_vecs, integrals, perturbations, eps_mu=eps, eps_resp=eps, triplet=True, overwrite=False)

for k, (i,j) in enumerate(nuc_pair):
    label1 = f'FC_{i}'
    label2 = f'FC_{j}'
    e11_fc[k,0,0] = e11_fc[k,1,1] = e11_fc[k,2,2] = -response_functions[(label1, label2, 0.0)]


print('SSCC (in Hz):')
j_tensor_fc = convert_unit(e11_fc, mol, active_atoms)
j_tensor_sd = convert_unit(e11_sd, mol, active_atoms)
j_tensor_pso = convert_unit(e11_pso, mol, active_atoms)
j_tensor_dso = convert_unit(e11_dso, mol, active_atoms)
j_tensor_total = j_tensor_fc + j_tensor_sd + j_tensor_pso + j_tensor_dso
for (i,j) in nuc_pair:
    print(f'{i} {j}:  DSO={j_tensor_dso[i,j]:.6f}, PSO={j_tensor_pso[i,j]:.6f}, FC={j_tensor_fc[i,j]:.6f}, SD={j_tensor_sd[i,j]:.6f}, Total={j_tensor_total[i,j]:.6f}')

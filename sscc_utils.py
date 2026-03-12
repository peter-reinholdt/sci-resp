#!/usr/bin/env python

import numpy as np
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.data import nist

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

def pso_integrals(mol):
    pso_integrals_ao = []
    for ia in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(ia))
        h1ao = mol.intor_asymmetric('int1e_prinvxp', 3)
        for operator in h1ao.reshape(3, mol.nao, mol.nao):
                pso_integrals_ao.append(operator)
    return pso_integrals_ao

def sd_integrals(mol):
    sd_integrals_ao = []
    for ia in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(ia))
        a01p = nist.G_ELECTRON * 0.25 * mol.intor('int1e_sa01sp', 12).reshape(3,4,mol.nao,mol.nao)
        h1ao = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2))
        # remove FC from FC+SD integral
        coords = mol.atom_coord(ia).reshape(1, 3)
        ao = mol.eval_gto('GTOval', coords)
        fc = 8*np.pi/3 * np.einsum('ip,iq->pq', ao, ao) * (nist.G_ELECTRON/2) / 2
        h1ao -= np.einsum('xy,mn->xymn', np.eye(3), fc)
        for i, operator in enumerate(h1ao.reshape(9, mol.nao, mol.nao)):
            sd_integrals_ao.append(operator)
    return sd_integrals_ao

def fc_integrals(mol):
    fc_integrals_ao = []
    for ia in range(mol.natm):
        coords = mol.atom_coord(ia).reshape(1, 3)
        ao = mol.eval_gto('GTOval', coords)
        h1ao = 8*np.pi/3 * np.einsum('ip,iq->pq', ao, ao) * (nist.G_ELECTRON/2) / 2
        fc_integrals_ao.append(h1ao)
    return fc_integrals_ao

def atom_gyro_list(mol):
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

def convert_unit(e11, mol, nuc_pair):
    # unit conversions
    e11 = e11*nist.ALPHA**4
    nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2Hz = nist.HARTREE2J / nist.PLANCK
    unit = au2Hz * nuc_magneton ** 2
    iso_ssc = unit * np.einsum('kii->k', e11) / 3
    natm = mol.natm
    ktensor = np.zeros((natm,natm))
    for k, (i, j) in enumerate(nuc_pair):
        ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
    gyro = atom_gyro_list(mol)
    jtensor = np.einsum('ij,i,j->ij', ktensor, gyro, gyro)
    return jtensor



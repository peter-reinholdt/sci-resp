#!/usr/bin/env python

import numpy as np

def davidson_response(A, b, hdiag, tol=1e-3, maxiter=100, verbose=False, guess=None):
    if np.allclose(b, 0.0, atol=1e-20):
        return b

    def matvec(v, A=A):
        if np.allclose(v.imag, 0.):
            Av = A(v.real)
        else:
            Av_real = A(v.real)
            Av_imag = A(v.imag)
            Av = Av_real + 1j*Av_imag
        return Av
    if guess is None:
        diagonal_guess = b/(hdiag)
        diagonal_guess /= np.linalg.norm(diagonal_guess)
        guess = diagonal_guess
    V = guess.astype(np.complex128).reshape(-1, 1)
    AV = np.zeros_like(V, dtype=np.complex128)
    AV[:, 0] = matvec(V[:, 0])

    bred = V.T @ b
    for i in range(maxiter):
        Ered = V.T @ AV
        xred = np.linalg.solve(Ered, bred)
        x = xred @ V.T
        residual = b - (xred @ AV.T)
        if verbose:
            print(f'Iteration {i+1:3d} residual norm: {np.linalg.norm(residual):6e}')
        if np.linalg.norm(residual) < tol:
            return x
        delta = residual/(hdiag)
        delta = delta/np.linalg.norm(delta)
        vnew = delta - V @ (V.T @ delta)
        vnew = vnew/np.linalg.norm(vnew)
        V = np.hstack([V, vnew[:, None]])
        bred = V.T @ b
        AV = np.hstack([AV, matvec(vnew)[:, None]])
    raise ValueError('Not converged')

def solve_ci(hvp, hdiag, roots, tol=1e-6, maxiter=100, verbose=False, c0=None):
    """
    Solves for eigenvalues and eigenvectors of a hessian.

    Args:
        hvp (callable): hessian-vector product, function that implements the matrix-vector product of the hessian with a trial vector
        hdiag (array): (approximate) diagonal hessian elements
        roots (int): number of roots to solve for
        tol (float): convergence tolerance on the residual norm
    """
    # select initial unit vectors based on diagonal hessian
    dim = len(hdiag)
    if c0 is not None:
        print(c0.shape)
        assert c0.shape[0] == dim
        V = c0
    else:
        V = np.zeros((dim, roots))
        V[np.argsort(hdiag)[:roots], np.arange(roots)] = 1.0

    AV = hvp(V)
    for i in range(maxiter):
        S = V.T @ AV
        L, Z = np.linalg.eigh(S)
        L = L[:roots]
        Z = Z[:, :roots]
        X = V @ Z
        AX = AV @ Z
        r = (AX - L[None, :]*X)
        if verbose:
            for k in range(len(L)):
                print(f'{i+1}       {k+1}   {L[k]:.6e}    {np.linalg.norm(r[:,k]):.6e}')
        if (all(np.linalg.norm(r, axis=0) < tol)):
            return L, X
        theta = 1e-3
        denom = L[:,None] - hdiag
        denom[np.abs(denom)<theta] = theta
        delta = np.array([1/(denom[k,:])*r[:, k] for k in range(len(L))]).T
        delta_norms = np.linalg.norm(delta, axis=0)
        delta /= delta_norms

        new_vs = []
        for k in range(len(L)):
            q = delta[:, k] - V @ (V.T @ delta[:, k])
            qnorm = np.linalg.norm(q)
            residual_norm = np.linalg.norm(r[:,k])
            if qnorm > 1e-3 and residual_norm > tol:
                vp = q/qnorm
                V = np.hstack([V, vp[:, None]])
                new_vs.append(vp)
        new_vs = np.array(new_vs).T
        AV = np.hstack([AV, hvp(new_vs)])
    raise ValueError('Not converged')

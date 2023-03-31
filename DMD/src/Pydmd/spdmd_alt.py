# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 01:55:32 2020

@author: AdminF
"""

from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag
from numpy.linalg import inv, eig, pinv, norm, solve, cholesky
from scipy.linalg import svd, svdvals
from scipy.sparse import csc_matrix as sparse
from scipy.sparse import vstack as spvstack
from scipy.sparse import hstack as sphstack
from scipy.sparse.linalg import spsolve
from functools import partial
from typing import NamedTuple
from dataclasses import dataclass
from multiprocessing import Pool
from Pydmd.dmd import DMD

@dataclass
class spdmd_ans:
    gamma = None 
    Nz = None
    Jsp = None
    Jpol = None
    Ploss = None
    xsp = None
    xpol = None
    idx = None

def admm_for_dmd_simple(P, q, s, gamma_vec, rho=1, maxiter=10000, eps_abs=1e-6, eps_rel=1e-4):

    # blank return value
    answer = type('ADMMAnswer', (object,), {})()
    
    # check input vars
    P = np.squeeze(P)
    q = np.squeeze(q)[:,np.newaxis]
    gamma_vec = np.squeeze(gamma_vec)
    if P.ndim != 2:
        raise ValueError('invalid P')
    if q.ndim != 2:
        raise ValueError('invalid q')
    if gamma_vec.ndim != 1:
        raise ValueError('invalid gamma_vec')

    # number of optimization variables
    n = len(q)

    # identity matrix
    I = np.eye(n)

    # allocate memory for gamma-dependent output variables
    answer.gamma = gamma_vec
    answer.Nz    = np.zeros([len(gamma_vec),]) # number of non-zero amplitudes
    answer.Jsp   = np.zeros([len(gamma_vec),]) # square of Frobenius norm (before polishing)
    answer.Jpol  = np.zeros([len(gamma_vec),]) # square of Frobenius norm (after polishing)
    answer.Ploss = np.zeros([len(gamma_vec),]) # optimal performance loss (after polishing)
    answer.xsp   = np.zeros([n, len(gamma_vec)], dtype='complex') # vector of amplitudes (before polishing)
    answer.xpol  = np.zeros([n, len(gamma_vec)], dtype='complex') # vector of amplitudes (after polishing)
    answer.idx = None
    # Cholesky factorization of matrix P + (rho/2)*I
    Prho = P + (rho/2) * I
    Plow = cholesky(Prho)
    Plow_star = Plow.conj().T

    # sparse P (for KKT system)
    Psparse = sparse(P)

    for i, gamma in enumerate(gamma_vec):

        # initial conditions
        y = np.zeros([n, 1], dtype='complex') # Lagrange multiplier
        z = np.zeros([n, 1], dtype='complex') # copy of x

        # Use ADMM to solve the gamma-parameterized problem  
        for step in range(maxiter):

            # x-minimization step
            u = z - (1/rho) * y
            # x = solve((P + (rho/2) * I), (q + rho * u))
            xnew = solve(Plow_star, solve(Plow, q + (rho/2) * u))

            # z-minimization step       
            a = (gamma/rho) * np.ones([n, 1])
            v = xnew + (1/rho) * y
            # soft-thresholding of v
            znew = multiply(multiply(np.divide(1 - a, np.abs(v)), v), (np.abs(v) > a))

            # primal and dual residuals
            res_prim = norm(xnew - znew, 2)
            res_dual = rho * norm(znew - z, 2)

            # Lagrange multiplier update step
            y += rho * (xnew - znew)

            # stopping criteria
            eps_prim = np.sqrt(n) * eps_abs + eps_rel * np.max([norm(xnew, 2), norm(znew, 2)])
            eps_dual = np.sqrt(n) * eps_abs + eps_rel * norm(y, 2)

            if (res_prim < eps_prim) and (res_dual < eps_dual):
                break
            else:
                z = znew        

        # record output data
        answer.xsp[:,i] = z.squeeze() # vector of amplitudes
        answer.Nz[i] = np.count_nonzero(answer.xsp[:,i]) # number of non-zero amplitudes
        answer.Jsp[i] = (
            np.real(dot(dot(z.conj().T, P), z))
            - 2 * np.real(dot(q.conj().T, z))
            + s) # Frobenius norm (before polishing)

        # polishing of the nonzero amplitudes
        # form the constraint matrix E for E^T x = 0
        ind_zero = np.flatnonzero(np.abs(z) < 1e-12) # find indices of zero elements of z
        m = len(ind_zero) # number of zero elements

        if m > 0:

            # form KKT system for the optimality conditions
            E = I[:,ind_zero]
            E = sparse(E, dtype='complex')
            KKT = spvstack([
                sphstack([Psparse, E], format='csc'),
                sphstack([E.conj().T, sparse((m, m), dtype='complex')], format='csc'),
                ], format='csc')            
            rhs = np.vstack([q, np.zeros([m, 1], dtype='complex')]) # stack vertically

            # solve KKT system
            sol = spsolve(KKT, rhs)
        else:
            sol = solve(P, q)

        # vector of polished (optimal) amplitudes
        xpol = sol[:n]

        # record output data
        answer.xpol[:,i] = xpol.squeeze()

        # polished (optimal) least-squares residual
        answer.Jpol[i] = (
            np.real(dot(dot(xpol.conj().T, P), xpol))
            - 2 * np.real(dot(q.conj().T, xpol))
            + s)

        # polished (optimal) performance loss 
        answer.Ploss[i] = 100 * np.sqrt(answer.Jpol[i]/s)
        if step % 10 == 0:
            print(f"idx No. {i} has run {step} iterations")
        if step % 10 == 0:
            print(f"idx No. {i} has run {step} iterations")
    print('done')
    return answer

def admm_for_dmd_par(P, q, s, gamma_vec, rho=1, maxiter=10000, eps_abs=1e-6, eps_rel=1e-4, nproc=2):

    # blank return value
    # answer = type('ADMMAnswer', (object,), {})()

    # check input vars
    print('admm multiprocessing running...')
    if isinstance(gamma_vec, list):
        nn = len(gamma_vec)
    elif isinstance(gamma_vec, np.ndarray):
        nn = gamma_vec.shape[0]
    else:
        raise TypeError("wrong error of gamma, please check")

    print(f"total numbers of gamma to optimize: {len(gamma_vec)}")
    P = np.squeeze(P)
    q = np.squeeze(q)[:,np.newaxis]
    gamma_vec = np.squeeze(gamma_vec)
    if P.ndim != 2:
        raise ValueError('invalid P')
    if q.ndim != 2:
        raise ValueError('invalid q')
    if gamma_vec.ndim != 1:
        raise ValueError('invalid gamma_vec')

    # number of optimization variables
    n = len(q)

    # identity matrix
    I = np.eye(n)
    # result = [answer]*n
    # allocate memory for gamma-dependent output variables
    # answer.gamma = gamma_vec
    # answer.Nz    = np.zeros([len(gamma_vec),]) # number of non-zero amplitudes
    # answer.Jsp   = np.zeros([len(gamma_vec),]) # square of Frobenius norm (before polishing)
    # answer.Jpol  = np.zeros([len(gamma_vec),]) # square of Frobenius norm (after polishing)
    # answer.Ploss = np.zeros([len(gamma_vec),]) # optimal performance loss (after polishing)
    # answer.xsp   = np.zeros([n, len(gamma_vec)], dtype='complex') # vector of amplitudes (before polishing)
    # answer.xpol  = np.zeros([n, len(gamma_vec)], dtype='complex') # vector of amplitudes (after polishing)
    # answer.idx = None
    # Cholesky factorization of matrix P + (rho/2)*I
    Prho = P + (rho/2) * I
    Plow = cholesky(Prho)
    Plow_star = Plow.conj().T

    # sparse P (for KKT system)
    # recons_idx = [i for i in range(n)]

    Psparse = sparse(P)
    # for i, (gamma, idx) in enumerate(zip(gamma_vec, recons_idx)):
    #     result[i].recons_idx = idx
    #     result[i].gamma = gamma
    # print(result[0].recons_idx)
    # answer = type('ADMMAnswer', (object,), {})()

    result_ans = admm_for_dmd_MP(gamma_vec,
                            P, s, q, Psparse, maxiter, rho, Plow_star, 
                            Plow, eps_abs, eps_rel,
                            nproc=nproc)
    return_result = spdmd_ans()
    cls_attr = [a for a in dir(result_ans[0]) if not a.startswith('__')]
    # print(cls_attr)
    for attr in cls_attr:
        # print(attr)
        result_ans[0].__getattribute__(attr)
        value_tmp = np.asarray([result_ans[i].__getattribute__(attr) for i in range(len(result_ans))]).squeeze().T
        return_result.__setattr__(attr, value_tmp)

    return return_result



def admm_for_dmd_MP(gamma_val,
                    P, s, q, Psparse, maxiter, rho, Plow_star, Plow, eps_abs, eps_rel,
                    nproc=2):

    func_MP = partial(admm_for_dmd_func, P, s, q, Psparse, maxiter, rho, Plow_star, Plow, eps_abs, eps_rel)
    # answer = type('ADMMAnswer', (object,), {})()
    # n = len(q)
    recons_idx = np.arange(0, len(gamma_val), dtype=np.int)
    # print(recons_idx)
    input_arg = zip(gamma_val, recons_idx)
    pool = Pool(nproc)
    result_ans = pool.starmap(func_MP, input_arg)
    return result_ans

def admm_for_dmd_func(P, s, q, Psparse, maxiter, rho, Plow_star, Plow, eps_abs, eps_rel, 
                      gamma_val, recons_idx):
    
    n = len(q)
    # identity matrix
    I = np.eye(n)
    # answer = type('ADMMAnswer', (object,), {})()
    answer = spdmd_ans()
    answer.recons_idx = recons_idx
    answer.gamma = gamma_val
    answer.Nz    = np.zeros([1,]) # number of non-zero amplitudes
    answer.Jsp   = np.zeros([1,]) # square of Frobenius norm (before polishing)
    answer.Jpol  = np.zeros([1,]) # square of Frobenius norm (after polishing)
    answer.Ploss = np.zeros([1,]) # optimal performance loss (after polishing)
    answer.xsp   = np.zeros([n, 1], dtype='complex') # vector of amplitudes (before polishing)
    answer.xpol  = np.zeros([n, 1], dtype='complex') # vector of amplitudes (after polishing)
    
    if not isinstance(gamma_val,(list, np.ndarray)):
        gamma_val = [gamma_val]

    if not isinstance(recons_idx, (list, np.ndarray)):
        recons_idx = [recons_idx]

    zip_iter = zip(gamma_val, recons_idx)
    # print(type(gamma_vec))
    # print(type(recons_idx))
  
    for i, (gamma, idx) in enumerate(zip_iter):
    # initial conditions
        y = np.zeros([n, 1], dtype='complex') # Lagrange multiplier
        z = np.zeros([n, 1], dtype='complex') # copy of x

        # Use ADMM to solve the gamma-parameterized problem  
        for step in range(maxiter):

            # x-minimization step
            u = z - (1/rho) * y
            # x = solve((P + (rho/2) * I), (q + rho * u))
            xnew = solve(Plow_star, solve(Plow, q + (rho/2) * u))

            # z-minimization step       
            a = (gamma/rho) * np.ones([n, 1])
            v = xnew + (1/rho) * y
            # soft-thresholding of v
            znew = multiply(multiply(np.divide(1 - a, np.abs(v)), v), (np.abs(v) > a))

            # primal and dual residuals
            res_prim = norm(xnew - znew, 2)
            res_dual = rho * norm(znew - z, 2)

            # Lagrange multiplier update step
            y += rho * (xnew - znew)

            # stopping criteria
            eps_prim = np.sqrt(n) * eps_abs + eps_rel * np.max([norm(xnew, 2), norm(znew, 2)])
            eps_dual = np.sqrt(n) * eps_abs + eps_rel * norm(y, 2)

            if (res_prim < eps_prim) and (res_dual < eps_dual):
                break
            else:
                z = znew        

        # record output data
        answer.xsp[:,i] = z.squeeze() # vector of amplitudes
        answer.Nz[i] = np.count_nonzero(answer.xsp[:,i]) # number of non-zero amplitudes
        answer.Jsp[i] = (
            np.real(dot(dot(z.conj().T, P), z))
            - 2 * np.real(dot(q.conj().T, z))
            + s) # Frobenius norm (before polishing)

        # polishing of the nonzero amplitudes
        # form the constraint matrix E for E^T x = 0
        ind_zero = np.flatnonzero(np.abs(z) < 1e-12) # find indices of zero elements of z
        m = len(ind_zero) # number of zero elements

        if m > 0:

            # form KKT system for the optimality conditions
            E = I[:,ind_zero]
            E = sparse(E, dtype='complex')
            KKT = spvstack([
                sphstack([Psparse, E], format='csc'),
                sphstack([E.conj().T, sparse((m, m), dtype='complex')], format='csc'),
                ], format='csc')            
            rhs = np.vstack([q, np.zeros([m, 1], dtype='complex')]) # stack vertically

            # solve KKT system
            sol = spsolve(KKT, rhs)
        else:
            sol = solve(P, q)

        # vector of polished (optimal) amplitudes
        xpol = sol[:n]

        # record output data
        answer.xpol[:,i] = xpol.squeeze()

        # polished (optimal) least-squares residual
        answer.Jpol[i] = (
            np.real(dot(dot(xpol.conj().T, P), xpol))
            - 2 * np.real(dot(q.conj().T, xpol))
            + s)

        # polished (optimal) performance loss 
        answer.Ploss[i] = 100 * np.sqrt(answer.Jpol[i]/s)
        if idx % 100 == 0 or step>6000:
            print(f"idx {idx} done by  {step} iterations")
        # if i % 5 == 0:
        # if idx % 10 == 0:
        # print(f"idx {idx} is done")
        # print('done')
        return answer

class SpDMD(DMD):
    
    def __init__(self,
                 svd_rank=0,
                 tlsq_rank=0,
                 exact=False,
                 opt=True):
        super(SpDMD, self).__init__(svd_rank, tlsq_rank, exact, opt)
        self.b_opt = None
        self.admm_result = None
        self.P = None
        self.q = None 
        self.s = None
        
    # def fit(self, X):
    #     """
    #     Compute the Dynamic Modes Decomposition to the input data.

    #     :param X: the input snapshots.
    #     :type X: numpy.ndarray or iterable
    #     """
    #     self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

    #     n_samples = self._snapshots.shape[1]
    #     X = self._snapshots[:, :-1]
    #     Y = self._snapshots[:, 1:]

    #     X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

    #     U, s, V = self._compute_svd(X, self.svd_rank)

    #     self._Atilde = self._build_lowrank_op(U, s, V, Y)

    #     self._svd_modes = U

    #     self._eigs, self._modes = self._eig_from_lowrank_op(
    #         self._Atilde, Y, U, s, V, self.exact)

    #     # Default timesteps
    #     self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
    #     self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

    #     self._b = self._compute_amplitudes(self._modes, self._snapshots,
    #                                         self._eigs, self.opt)


    def admm_opt(self, gamma_vec,
                    MP=True, nproc=2,
                    rho=1, maxiter=10000, eps_abs=1e-6, eps_rel=1e-4):
        
        if self.P is None or self.q is None:
            # compute the vandermonde matrix
            omega = old_div(np.log(self.eigs), self.original_time['dt'])
            vander = np.exp(
                np.multiply(*np.meshgrid(omega, self.dmd_timesteps))).T

            # perform svd on all the snapshots
            U, s, V = np.linalg.svd(self._snapshots, full_matrices=False)

            self.P = np.multiply(
                np.dot(self.modes.conj().T, self.modes),
                np.conj(np.dot(vander, vander.conj().T)))
            tmp = (np.dot(np.dot(U, np.diag(s)), V)).conj().T
            self.q = np.conj(np.diag(np.dot(np.dot(vander, tmp), self.modes)))
                    #  np.conj(diag(dot(dot(Vand, (dot(dot(U, diag(sv)), Vh)).conj().T), Phi)))
            self.s = norm(diag(s), ord='fro')**2
            self._b_opt = np.linalg.solve(self.P, self.q)
        if MP:
            self.admm_result = admm_for_dmd_par(self.P, self.q, self.s, gamma_vec, 
                                                rho=rho, maxiter=maxiter, eps_abs=eps_abs, 
                                                eps_rel=eps_rel, nproc=nproc)
        else:
            self.admm_result = admm_for_dmd_simple(self.P, self.q, self.s, gamma_vec, 
                                                    rho=rho, maxiter=maxiter, 
                                                    eps_abs=eps_abs, eps_rel=eps_rel)
        return self
import numpy as np
import scipy as sp

def admm(z, y, gamma, q, Prho, n, 
        rho=1, max_admm_iter=10000, eps_abs=1e-6, eps_rel=1e-4):
    """Alternating direction method of multipliers."""
    # Optimization:
    # This has been reasonably optimized already and performs ~3x
    # faster than a naive translation of the matlab version.

    # Two major changes are a custom function for calculating
    # the norm of a 1d vector and accessing the lapack solver
    # directly.

    # However it still isn't as fast as matlab (~1/3rd the
    # speed).

    # There are two complexity sources:
    # 1. the matrix solver. I can't see how this can get any
    #    faster (tested with Intel MKL on Canopy).
    # 2. the test for convergence. This is the dominant source
    #    now (~3x the time of the solver)

    # One simple speedup (~2x faster) is to only test
    # convergence every n iterations (n~10). However this breaks
    # output comparison with the matlab code. This might not
    # actually be a problem.

    # Further avenues for optimization:
    # - write in cython and import as compiled module, e.g.
    #   http://docs.cython.org/src/userguide/numpy_tutorial.html
    # - use two cores, with one core performing the admm and
    #   the other watching for convergence.

    a = (gamma / rho)
    # q = self.dmd.q
    print('precompute cholesky decomposition')
    # print(Prho)
    # precompute cholesky decomposition
    # C = np.linalg.cholesky(Prho, lower=False)
    C =  np.linalg.cholesky(Prho).T.conj()
    # link directly to LAPACK fortran solver for positive
    # definite symmetric system with precomputed cholesky decomp:
    print('definite symmetric system with precomputed cholesky decomp')
    potrs, = sp.linalg.get_lapack_funcs(('potrs',), arrays=(C, q))
    print('simple norm of a 1d vector, called directly from BLAS')
    # simple norm of a 1d vector, called directly from BLAS
    norm, = sp.linalg.get_blas_funcs(('nrm2',), arrays=(q,))

    # square root outside of the loop
    root_n = np.sqrt(n)

    for ADMMstep in range(max_admm_iter):
        # print(f"iter{ADMMstep}")
        # if ADMMstep % 100 == 0:
        #     print(f"ADMMstep: {ADMMstep}")
        # ## x-minimization step (alpha minimisation)
        u = z - (1. / rho) * y
        qs = q + (rho / 2.) * u
        # Solve P x = qs, using fact that P is hermitian and
        # positive definite and assuming P is well behaved (no
        # inf or nan).
        xnew = potrs(C, qs, lower=False, overwrite_b=False)[0]
        # ##

        # ## z-minimization step (beta minimisation)
        v = xnew + (1 / rho) * y
        # Soft-thresholding of v
        # zero for |v| < a
        # v - a for v > a
        # v + a for v < -a
        # n.b. This doesn't actually do this because v is
        # complex. This is the same as the matlab source. You might
        # want to use np.sign, but this won't work because v is complex.
        abs_v = np.abs(v)
        znew = ((1 - a / abs_v) * v) * (abs_v > a)
        # ##

        # ## Lagrange multiplier update step
        y = y + rho * (xnew - znew)
        # ##

        # ## Test convergence of admm
        # Primal and dual residuals
        res_prim = norm(xnew - znew)
        res_dual = rho * norm(znew - z)

        # Stopping criteria
        eps_prim = root_n * eps_abs \
                    + eps_rel * max(norm(xnew), norm(znew))
        eps_dual = root_n * eps_abs + eps_rel * norm(y)

        if (res_prim < eps_prim) & (res_dual < eps_dual):
            return z
        else:
            z = znew
    return z

def KKT_solve(n, q, P, z):
    """Polishing of the sparse vector z. Seeks solution to
    E^T z = 0
    """
    # indices of zero elements of z (i.e. amplitudes that
    # we are ignoring)
    ind_zero = abs(z) < 1E-12

    # number of zero elements
    m = ind_zero.sum()

    # Polishing of the nonzero amplitudes
    # Form the constraint matrix E for E^T x = 0
    E = np.identity(n)[:, ind_zero]
    # n.b. we don't form the sparse matrix as the original
    # matlab does as it doesn't seem to affect the
    # computation speed or the output.
    # If you want to use a sparse matrix, use the
    # scipy.sparse.linalg.spsolve solver with a csc matrix
    # and stack using scipy.sparse.{hstack, vstack}

    # Form KKT system for the optimality conditions
    KKT = np.vstack((np.hstack((P, E)),
                        np.hstack((E.T.conj(), np.zeros((m, m))))
                        ))
    rhs = np.hstack((q, np.zeros(m)))

    # Solve KKT system
    return sp.linalg.solve(KKT, rhs)

def residuals(P, q, sp_s, x):
    """Calculate the residuals from a minimised
    vector of amplitudes x.
    """
    # conjugate transpose
    x_ = x.T.conj()
    q_ = q.T.conj()

    x_P = np.dot(x_, P)
    x_Px = np.dot(x_P, x)
    q_x = np.dot(q_, x)

    return x_Px.real - 2 * q_x.real + sp_s
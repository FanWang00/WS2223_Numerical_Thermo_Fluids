import numpy as np
import scipy.linalg as linalg
from .spdmd import SPDMD_recons
from .admm import admm, residuals, KKT_solve
from functools import partial
# from .dmd import DMD
# from .util import to_data
from multiprocessing import Pool


def MP_optimize_gamma(n, P, q, sp_s, Prho, gamma):
    MP_func = partial(optimize_gamma, n, P, q, sp_s, Prho)
    MP_idx = [i for i in range(len(gamma))]
    pool = Pool(4)
    ret = pool.starmap(MP_func, zip(gamma, MP_idx))
    pool.terminate()
    pool.join()
    return ret

def optimize_gamma(n, P, q, sp_s, Prho, gamma, MP_idx=None):
    """Minimise
        J(a)
    subject to
        E^T a = 0

    This amounts to finding the optimal amplitudes for a given
    sparsity. Sparsity is encoded in the structure of E.

    The first step is solved using ADMM.

    The second constraint is satisfied using KKT_solve.
    """
    if MP_idx is not None:
        print(f"process {MP_idx} running...")
    # Use ADMM to solve the gamma-parameterized problem,
    # minimising J, with initial conditions z0, y0
    y0 = np.zeros(n)  # Lagrange multiplier
    z0 = np.zeros(n)  # initial amplitudes
    print(f"entering admm")
    z = admm(z0, y0, gamma, q, Prho, n)

    # Now use the minimised amplitudes as the input to the
    # sparsity contraint to create a vector of polished
    # (optimal) amplitudes
    xpol = KKT_solve(n, q, P, z)[:n]

    # outputs that we care about...
    # vector of amplitudes
    sparse_amplitudes = z
    # number of non-zero amplitudes
    num_nonzero = (z != 0).sum()
    
    # least squares residual
    sp_residuals = residuals(P, q, sp_s, z)

    # Vector of polished (optimal) amplitudes
    polished_amplitudes = xpol
    # Polished (optimal) least-squares residual
    polished_residual = residuals(P, q, sp_s, xpol)
    # print(polished_residual)
    # Polished (optimal) performance loss
    polished_performance_loss = 100 * \
        np.sqrt(polished_residual / sp_s)
    if MP_idx is not None:
        print(f"process {MP_idx} is done")

    return {'xsp':   sparse_amplitudes,
            'Nz':    num_nonzero,
            'Jsp':   sp_residuals,
            'xpol':  polished_amplitudes,
            'Jpol':  polished_residual,
            'Ploss': polished_performance_loss,
            }


class SparseDMD(object):
    def __init__(self, snapshots=None, dmd=None, axis=-1, dt=1,
                 rho=1, maxiter=10000, eps_abs=1e-6, eps_rel=1e-4):
        # TODO: allow data, axis as an argument instead of snapshots
        """Sparse Dynamic Mode Decomposition, using ADMM to find a
        sparse set of optimal dynamic mode amplitudes

        Inputs:
            snapshots - the matrix of data snapshots, shape (d, N)
                        where N is the number of snapshots and d is
                        the number of data points in a snapshot.

                        Alternately, multi-dimensional data can be
                        given here and it will be reshaped into the
                        snapshot matrix along the given `axis`.

            dmd     - optional precomputed DMD instance

            axis    - decomposition axis, default -1

            rho     - augmented Lagrangian parameter
            maxiter - maximum number of ADMM iterations
            eps_abs - absolute tolerance for ADMM
            eps_rel - relative tolerance for ADMM

        Defaults:
            If snapshots is not supplied and you have precomputed
            the dmd reduction [U^*X1, S, V], you can initialise the
            dmd with SparseDMD.dmd.init(U^*X1, S, V).

            rho = 1
            maxiter = 10000
            eps_abs = 1.e-6
            eps_rel = 1.e-4
        """
        self.rho = rho
        # self.max_admm_iter = maxiter
        # self.eps_abs = eps_abs
        # self.eps_rel = eps_rel

        if dmd is None:
            print("please input perform dmd first!")
        else:
            self.dmd = dmd
        # if snapshots is not None:
        #     self.dmd = DMD(snapshots, axis=axis, dt=dt)
        #     self.dmd.compute()
        # elif not snapshots and dmd is not None:
        #     self.dmd = dmd
        # elif not snapshots and not dmd:
        #     self.dmd = DMD()

    def compute_sparse(self, gammaval, MP=False):
        """Compute the sparse dmd structure and set as attribute."""
        self.gammaval = gammaval
        self.sparse = self.dmdsp(gammaval, MP=MP)

    def dmdsp(self, gammaval, MP=False):
        """Inputs:
            gammaval - vector of gamma to perform sparse optimisation over

        Returns:
            answer - gamma-parameterized structure containing
                answer.gamma - sparsity-promoting parameter gamma
                answer.xsp   - vector of amplitudes resulting from (SP)
                answer.xpol  - vector of amplitudes resulting from (POL)
                answer.Jsp   - J resulting from (SP)
                answer.Jpol  - J resulting from (POL)
                answer.Nz    - number of nonzero elements of x
                answer.Ploss - optimal performance loss 100*sqrt(J(xpol)/J(0))

        Additional information:

        http://www.umn.edu/~mihailo/software/dmdsp/
        """
        # Number of optimization variables
        self.n = len(self.dmd.q)

        # length of parameter vector
        ng = len(gammaval)

        Prho = self.dmd.P + (self.rho / 2.) * np.identity(self.n)

        answer = SparseAnswer(self.n, ng)
        answer.gamma = gammaval
        if MP:
            # print("MP  mode")
            # MP_func = partial(self.optimize_gamma, 
            #                     self.n, self.dmd.P, 
            #                     self.dmd.q, self.dmd.sp_s, Prho)
            # pool = Pool(2)
            # ret = pool.map(MP_func, gammaval)
            # pool.terminate()
            # pool.join()
            ret = MP_optimize_gamma(self.n, self.dmd.P, 
                                    self.dmd.q, self.dmd.sp_s, Prho,
                                    gammaval)
            for i, r in enumerate(ret):
                answer.xsp[:, i] = r['xsp']
                answer.xpol[:, i] = r['xpol']
                answer.Nz[i] = r['Nz']
                answer.Jsp[i] = r['Jsp']
                answer.Jpol[i] = r['Jpol']
                answer.Ploss[i] = r['Ploss']
        else:
            for i, gamma in enumerate(gammaval):
                ret = optimize_gamma(
                                self.n, self.dmd.P, 
                                self.dmd.q, self.dmd.sp_s,Prho,
                                gamma)

                answer.xsp[:, i] = ret['xsp']
                answer.xpol[:, i] = ret['xpol']

                answer.Nz[i] = ret['Nz']
                answer.Jsp[i] = ret['Jsp']
                answer.Jpol[i] = ret['Jpol']
                answer.Ploss[i] = ret['Ploss']

        answer.nonzero[:] = answer.xsp != 0

        return answer

    def reconstruction(self, Ni):
        """Compute a reconstruction of the input data based on a sparse
        selection of modes.

        Ni - the index that selects the desired number of
             modes in self.sparse.Nz

        shape - the shape of the original input data. If not supplied,
                the original snapshots will be reconstructed.

        Returns a SparseReconstruction with the following attributes:

        r.nmodes  # number of modes (3)
        r.data    # the original data
        r.rdata   # the reconstructed data (or snapshots)
        r.modes   # the modes (3 of them)
        r.freqs   # corresponding complex frequencies
        r.amplitudes  # corresponding amplitudes
        r.ploss   # performance loss
        """
        return SPDMD_recons(self,
                            number_index=Ni)
        # return SparseReconstruction(self,
        #                             number_index=Ni,
        #                             shape=None,
        #                             axis=None)


class SparseAnswer(object):
    """A set of results from sparse dmd optimisation.

    Attributes:
    gamma     the parameter vector
    nz        number of non-zero amplitudes
    nonzero   where modes are nonzero
    jsp       square of frobenius norm (before polishing)
    jpol      square of frobenius norm (after polishing)
    ploss     optimal performance loss (after polishing)
    xsp       vector of sparse amplitudes (before polishing)
    xpol      vector of amplitudes (after polishing)
    """
    def __init__(self, n, ng):
        """Create an empty sparse dmd answer.

        n - number of optimization variables
        ng - length of parameter vector
        """
        # the parameter vector
        self.gamma = np.zeros(ng)
        # number of non-zero amplitudes
        self.Nz = np.zeros(ng)
        # square of frobenius norm (before polishing)
        self.Jsp = np.zeros(ng, dtype=np.complex)
        # square of frobenius norm (after polishing)
        self.Jpol = np.zeros(ng, dtype=np.complex)
        # optimal performance loss (after polishing)
        self.Ploss = np.zeros(ng, dtype=np.float64)
        # vector of amplitudes (before polishing)
        self.xsp = np.zeros((n, ng), dtype=np.complex)
        # vector of amplitudes (after polishing)
        self.xpol = np.zeros((n, ng), dtype=np.complex)

    @property
    def nonzero(self):
        """where amplitudes are nonzero"""
        return self.xsp != 0


class SparseReconstruction(object):
    """Reconstruction of the input data based on a
    desired number of modes.

    Returns an object with the following attributes:

        r = dmd.make_sparse_reconstruction(nmodes=3)

        r.nmodes  # number of modes (3)
        r.data    # the reconstructed data
        r.modes   # the modes (3 of them)
        r.freqs   # corresponding complex frequencies
        r.amplitudes  # corresponding amplitudes
        r.ploss   # performance loss

    Returns error if the given number of modes cannot be found
    over the gamma we've looked at.

    TODO: think about a gamma search function?
    """
    def __init__(self, sparse_dmd, number_index, shape=None, axis=-1):
        """
        sparse_dmd - a SparseDMD instance with the sparse solution computed

        number_index - the index that selects the desired number of
                       modes in sparse_dmd.sparse.Nz

        shape - the original input data shape. Used for reshaping the
                reconstructed snapshots.

        axis - the decomposition axis in the input data. Defaults to
               -1, i.e. will work with matrix of snapshots.
        """
        self.dmd = sparse_dmd.dmd
        self.sparse_dmd = sparse_dmd.sparse

        self.nmodes = self.sparse_dmd.Nz[number_index]
        self.Ni = number_index

        self.data_shape = shape
        self.axis = axis

        self.rmodes = self.sparse_reconstruction()

        nonzero = self.sparse_dmd.nonzero[:, number_index]
        self.nonzeros_test=nonzero
        # nonzero = np.where(nonzero, 1, 0)
        # print(f"nonzeros{nonzero}")
        self.modes = self.dmd.modes[:, nonzero]
        # self.freqs = self.dmd.Edmd*nonzero
        self.amplitudes = self.sparse_dmd.xpol[nonzero, number_index]
        self.ploss = self.sparse_dmd.Ploss[number_index]
        # self.recons_test = np.dot(self.modes, np.dot(self.amplitudes, self.dmd.sp_vander[nonzero,:])
        # self.modes = self.dmd.modes[:, nonzero]
        # self.freqs = self.dmd.Edmd[nonzero]
        # self.amplitudes = self.sparse_dmd.xpol[nonzero, number_index]
        # self.ploss = self.sparse_dmd.Ploss[number_index]

    def sparse_reconstruction(self):
        """Reconstruct the snapshots using a given number of modes.

        Ni is the index that gives the desired number of
        modes in `self.sparse.Nz`.

        shape is the shape of the original data. If None (default),
        the reconstructed snapshots will be returned; otherwise the
        snapshots will be reshaped to the original data dimensions,
        assuming that they were decomposed along axis `axis`.
        """
        amplitudes = np.diag(self.sparse_dmd.xpol[:, self.Ni])

        # print(amplitudes.shape)
        modes = self.dmd.modes
        time_series = self.dmd.sp_vander
        # we take the real part because for real data the modes are
        # in conjugate pairs and should cancel out. They don't
        # exactly because this is an optimal fit, not an exact
        # match.
        reconstruction = np.dot(modes, np.dot(amplitudes, time_series))
        return reconstruction.real

    # @property
    # def rdata(self):
    #     """Convenience function to return reduced modes reshaped
    #     into original data shape.
    #     """
    #     if self.data_shape is not None:
    #         data_reconstruction = to_data(self.rmodes,
    #                                       self.data_shape,
    #                                       self.axis)
    #         return data_reconstruction

    # @property
    # def dmodes(self):
    #     """Convenience function to return modes reshaped into original
    #     data shape.
    #     """
    #     return to_data(snapshots=self.modes,
    #                    shape=self.data_shape,
    #                    axis=self.axis)

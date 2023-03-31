
import numpy as np
# from sklearn.gaussian_process.kernels import RBF as rbf_kernel

from scipy.linalg import pinv2
# from past.utils import old_div
import copy
# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase
# from sklearn.metrics.pairwise import rbf_kernel

# from sklearn import gaussian_process
def pinv(x): return pinv2(x, rcond=10 * np.finfo(float).eps)

class KDMD(DMDBase):
    """ Kernel Dynamic Mode Decomposition (KDMD)

    Dynamically relevent dimensionality reduction using kernel-based methods
    to implicitly choose a larger subset of observable space than used by
    standard DMD.  This approach extracts a set of modes with fixed temoral
    behaviors (i.e., exponential growth or decay) and embeds the data
    in an approximate Koopman eigenfunction coordinate system.

    This implementation uses the numpy implementation of the singular value
    decomposition (SVD), and therefore requires the entire data set to fit in
    memory.  This algorithm runs in O(NM^2) time, where N is the size of a
    snapshot and M is the number of snapshots assuming N>M.

    Due to the similarities in implementation, this code computes four
    variants of Kernel DMD, only one of which has appeared in the literature.

    1) Kernel DMD (see Williams, Rowley, & Kevrekidis, 2015)
    2) Exact Kernel DMD (modes are based on the Y data rather than the X data)
    3) Total least squares kernel DMD (a combination of Williams 2014 and
        Hemati 2015)
    4) Exact, TLS, kernel DMD (a combination of Williams 2014 and Hemati 2015)


    Parameters
    ----------
    kernel_fun : function or functor (array, array) -> square array
        A kernel function that computes the inner products of data arranged
        in an array with snapshots along each *COLUMN* when the __call__
        method is evaluated.

    n_rank : int or None, optional
        Number of features to retain in the when performing DMD.  n_rank is
        an upper bound on the rank of the resulting DMD matrix.

        If n_rank is None (default), then n_snapshot modes will be retained.

    exact : bool, optional
        If false (default), compute the KDMD modes using the X data
        If true, compute the KDMD modes using the Y data

        See Tu et al., 2014 and Williams, Rowley, & Kevrekidis 2014
        for details.

    total : bool, optional
        If false (default), compute the standard KDMD modes
        If true, compute the total least squares KDMD modes

        See Hemati & Rowley, 2015 and Williams, Rowley,
        & Kevrekidis, 2015 for details.

    Attributes
    ----------
    evals : array, shape (n_rank,) or None
       The eigenvalues associated with each mode (None if not computed)

    modes: array, shape (n_dim, n_rank) or None
       The DMD modes associated with the eigenvalues in evals

    Phi : array, shape (n_rank, n_snapshots) or None
       An embedding of the X data

    Atilde : array, shape (n_rank, n_rank) or None
       The "KDMD matrix" used in mode computation

    Notes
    -----
    Implements the DMD algorithms as presented in:

    Williams, Rowley, and Kevrekidis.  A Kernel-Based Approach to
        Data-Driven Koopman Spectral Analysis, arXiv:1411.2260 (2014)

    Augmented with ideas from:

    Hemati and Rowley, De-biasing the dynamic mode decomposition
        for applied Koopman spectral analysis, arXiv:1502.03854 (2015).

    For kernel DMD as defined in Williams, exact=False and total=False
    """
    def __init__(self, kernel_fun, svd_rank=0, total=None, exact=False, opt=False, nx=None, ny=None,
                    n_sig=10):

        super(KDMD, self).__init__(svd_rank=svd_rank, tlsq_rank=0, exact=exact, opt=opt, nx=nx, ny=ny, n_sig=n_sig)
                    # self,          svd_rank=0, tlsq_rank=0, exact=False, opt=False, nx=None, ny=None,n_sig=10

    # def __init__(self, , n_rank=None, exact=False, total=False):
        self.kernel_fun = kernel_fun
        self.n_rank = svd_rank
        self.exact = exact
        self.total = total

        self._modes = None
        self._eigs = None
        self._Phi = None
        self._Atilde = None
        self._G = None
        self._A = None

    @property
    def modes(self):
        return self._modes

    @property
    def evals(self):
        return self._eigs

    @property
    def basis(self):
        return self._basis

    # @property
    # def Atilde(self):
    #     return self._Atilde

    def fit(self, X, Y=None):
        """ Fit a DMD model with the data in X (and Y)

        Parameters
        ----------
        X : array, shape (n_dim, n_snapshots)
            Data set where n_snapshots is the number of snapshots and
            n_dim is the size of each snapshot.  Note that spatially
            distributed data should be "flattened" to a vector.

            If Y is None, then the columns of X must contain a time-series
            of data taken with a fixed sampling interval.

        Y : array, shape (n_dim, n_snapsots)
            Data set containing the updated snapshots of X after a fixed
            time interval has elapsed.

        Returns
        -------
        self : object
            Returns this object containing the computed modes and eigenvalues
        """
        self._snapshots = copy.deepcopy(X)
        if Y is None:
            # Efficiently compute A, G, and optionally Gy
            # given a time series of data
            
            Gfull = self.kernel_fun(X, X)
            G = Gfull[:-1, :-1]
            A = Gfull[1:, :-1]

            if self.total:
                Gy = Gfull[1:, 1:]

            Y = X[:, 1:]
            X = X[:, :-1]
        else:  # Paired data

            try:
                gram_tuple = self.kernel_fun.compute_products(X, Y, self.total)
                # print(gram_tuple)
                if self.total:
                    G, A, Gy = gram_tuple
                else:
                    G, A = gram_tuple
                    # print(G.shape)
                    # print(A.shape)

            except AttributeError:
                G = self.kernel_fun(X, X)
                A = self.kernel_fun(Y, X)

                if self.total:
                    Gy = self.kernel_fun(Y, Y)

        # Rank is determined either by the specified value or
        # the number of snapshots
        # if self.n_rank is not None:
        if self.n_rank != 0:
            n_rank = min(self.n_rank, X.shape[1])
        else:
            n_rank = X.shape[1]

        # ====== Total Least Squares DMD: Project onto shared subspace ========
        if self.total:
            # Compute V using the method of snapshots

            sig2, V_stacked = np.linalg.eigh(G + Gy)
            inds = np.argsort(sig2)[::-1]  # sort by eigenvalue
            V_stacked = V_stacked[:, inds[:n_rank]]  # truncate to n_rank

            # Compute the "clean" data sets
            proj_Vh = V_stacked.dot(V_stacked.T)
            G = proj_Vh.dot(G).dot(proj_Vh)
            A = proj_Vh.dot(A).dot(proj_Vh)
            X = X.dot(proj_Vh)
            Y = Y.dot(proj_Vh)

        # ===== Kernel Dynamic Mode Decomposition Computation ======
        self._A = A
        self._G = G
        S2, U = np.linalg.eigh(G)
        # S2, U = np.linalg.eig(G)
        inds = np.argsort(S2)[::-1]
        U = U[:, inds[:n_rank]]
        S2 = S2[inds[:n_rank]]
        self._Atilde = U.T.dot(A).dot(U)/S2

        # Eigensolve gives modes and eigenvalues
        self._eigs, vecs = np.linalg.eig(self._Atilde)
        self._PhiX = (U.dot(vecs)).T

        # Two options: exact modes or projected modes
        if self.exact:
            PhiY = ((A.dot(U)/S2).dot(vecs)).T
            self._modes = Y.dot(np.linalg.pinv(PhiY))
        else:
            self._modes = X.dot(np.linalg.pinv(self._PhiX))
        ## amplitudes
        # self._b = np.linalg.lstsq(self._modes, self._snapshots.T[0], rcond=None)[0]
        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)
        # Default timesteps
        n_samples = self._snapshots.shape[1]
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}


        return self


class PolyKernel(object):
    """ Implements a simple polynomial kernel

    This class is meant as an example for implementing kernels.

    Parameters
    ----------
    alpha : int
        The power used in the polynomial kernel
    epsilon : double, optional
        Scaling parameter in the kernel, default is 1.

    Notes
    -----
    We refer to the transformation from state space to feature space a f 
    in all that follows.
    """

    def __init__(self, alpha=1, g_factor=1, g_gamma=1, kernel=1, epsilon=1.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.g_factor = g_factor
        self.kernel = kernel
        ## g_sigma: length scale of gaussian kernel 
        self.g_gamma = g_gamma
        
    def __call__(self, X, Y):
        """
        Compute the inner products (in feature space) of f(X)^T*f(Y)

        Parameters
        ----------
        X : array, shape (n_dim, n_snapshots)
            Data set where n_snapshots is the number of snapshots and
            n_dim is the size of each snapshot.  Note that spatially
            distributed data should be flattened to a vector.

        Y : array, shape (n_dim, n_snapsots)
            Data set containing the updated snapshots of X after a fixed
            time interval has elapsed.

        Returns
        -------
        self : array, shape (n_snapsots, n_snapshots)
            Returns the matrix of inner products in feature space
        """
        if self.kernel == 1:
            return (1.0 + X.T.dot(Y)/self.epsilon)**self.alpha
        if self.kernel == 2:
            # --> Import standard python packages
            from sklearn.metrics.pairwise import rbf_kernel
            # from sklearn.metrics.pairwise import rbf_kernel
            return rbf_kernel(X.T, Y.T, gamma=self.g_gamma)
            # return GaussianMatrix(X, Y, self.g_sigma)
            

    def compute_products(self, X, Y, Gy=False):
        """
        Compute the inner products f(X)^T*f(X), f(Y)^T*f(X), and if needed f(Y)^T*f(Y).

        For a polynomial kernel, this code is no more efficient than
        computing the terms individually.  Other kernels require
        knowledge of the complete data set, and must use this.

        Note: If this method is not implemented, the KDMD code will
        manually compute the inner products using the __call__ method.
        """

        if Gy:
            return self(X, X), self(Y, X), self(Y, Y)
        else:
            return self(X, X), self(Y, X)
        
def GaussianMatrix(X, Y, sigma):
    row,col=X.shape
    GassMatrix=np.zeros(shape=(col, col))
    # X=np.asarray(X)
    for i in range(col):
    # for v_i in X:
        # j = 0
#     for i in nb.prange(row):
#         v_i = X[i]
#         for j in nb.prange(row):
        # for v_j in Y:
        v_i = X[:, i]
        for j in range(col):
            v_j = X[:, j]
            GassMatrix[i ,j]=Gaussian_kernel(v_i.T, v_j.T, sigma)
            j+=1
        i+=1
    return GassMatrix

def Gaussian_kernel(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))
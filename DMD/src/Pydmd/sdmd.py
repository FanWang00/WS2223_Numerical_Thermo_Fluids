"""
Derived module from dmdbase.py for classic dmd.
"""

# --> Import standard python packages
import numpy as np
from scipy.linalg import pinv2
from past.utils import old_div
import copy
# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase

def pinv(x): return pinv2(x, rcond=10 * np.finfo(float).eps)


class SDMD(DMDBase):

    def __init__(self, max_rank=0, count=0, ngram=5, epsilon=1e-10,
                tlsq_rank=0, exact=False, opt=False, nx=None, ny=None,
                n_plot=10):
 
        super(SDMD, self).__init__(max_rank, tlsq_rank, exact, opt, nx, ny, n_plot)
        
        self.max_rank = max_rank
        self.count = 0
        self.ngram = ngram      # number of times to reapply Gram-Schmidt
        self.epsilon = epsilon  # tolerance for expanding the bases
        self._modes = None
        self.Qx = 0
        self.Qy = 0
        self.A = 0
        self.Gx = 0
        self.Gy = 0
        self._eigs = 0
        self._evecK = 0
    
    @property   
    def eigs(self):
        return self._eigs
    
    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.
        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray
        """
        omega = old_div(np.log(self.eigs), self.original_time['dt'])
        # omega = np.log(self.eigs) / self.original_time['dt']
        vander = np.exp(
            np.outer(omega, self.dmd_timesteps - self.original_time['t0']))
        return vander * self._b[:, None]

    def fit(self, X):
        
        # self._snapshots, self._snapshots_shape = self._col_major_2darray(X)
        self._snapshots = X
        # self._snapshots_shape = self._col_major_2darray(X)
        n_samples = self._snapshots.shape[1] - 1
        x = self._snapshots[:, :-1]
        y = self._snapshots[:, 1:]
        for i in range(n_samples):
            self.update(x[:, i].copy(), y[:, i].copy())
        self._eigendecomp()
        self._modes = self.Qx.dot(self._evecK)
        
        # self._b = self._compute_amplitudes(self._modes, self._snapshots,
        #                                    self._eigs, opt=False)
        self._b = np.linalg.lstsq(self._modes, self._snapshots.T[0], rcond=None)[0]
        # Default timesteps
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        return self
            
    
    def update(self, x_input, y_input):
        """Update the DMD computation with a pair of snapshots
        Add a pair of snapshots (x,y) to the data ensemble.  Here, if the
        (discrete-time) dynamics are given by z(n+1) = f(z(n)), then (x,y)
        should be measurements corresponding to consecutive states
        z(n) and z(n+1)
        """
    
        self.count += 1
        normx = np.linalg.norm(x_input)
        normy = np.linalg.norm(y_input)
        n = len(x_input)

        x = copy.deepcopy(x_input.reshape((n, 1)))
        y = copy.deepcopy(y_input.reshape((n, 1)))
        # process the first iterate
        if self.count == 1:
            # construct bases
            
            self.Qx = x / normx
            self.Qy = y / normy

            # compute
            self.Gx = normx**2
            self.Gy = normy**2
            self.A = normx * normy
            
            return

        # ---- Algorithm step 1 ----
        # classical Gram-Schmidt reorthonormalization

        rx = self.Qx.shape[1]
        ry = self.Qy.shape[1]
        

        xtilde = np.zeros((rx, 1))
        ytilde = np.zeros((ry, 1))
        

        ex = copy.deepcopy(x.reshape((n, 1)))
        ey = copy.deepcopy(y.reshape((n, 1)))

            
        for i in range(self.ngram):
            dx = self.Qx.T.dot(ex)
            dy = self.Qy.T.dot(ey)
            xtilde += dx
            ytilde += dy
            ex -= self.Qx.dot(dx)
            ey -= self.Qy.dot(dy)
            

        # ---- Algorithm step 2 ----
        # check basis for x and expand, if necessary

        if np.linalg.norm(ex) / normx > self.epsilon:
            # update basis for x
            # self.Qx = np.bmat([self.Qx, ex / np.linalg.norm(ex)])
            self.Qx = np.block([self.Qx, ex / np.linalg.norm(ex)])
            # increase size of Gx and A (by zero-padding)
            # self.Gx = np.bmat([[self.Gx, np.zeros((rx, 1))],
            #                    [np.zeros((1, rx+1))]])
            
            self.Gx = np.block([[self.Gx, np.zeros((rx, 1))],
                                [np.zeros((1, rx+1))]])
            # self.A = np.bmat([self.A, np.zeros((ry, 1))])
            self.A = np.block([self.A, np.zeros((ry, 1))])
            rx += 1

        # check basis for y and expand if necessary
        if np.linalg.norm(ey) / normy > self.epsilon:
            # update basis for y
            self.Qy = np.block([self.Qy, ey / np.linalg.norm(ey)])
            # increase size of Gy and A (by zero-padding)
            self.Gy = np.block([[self.Gy, np.zeros((ry, 1))],
                               [np.zeros((1, ry+1))]])
            self.A = np.block([[self.A],
                              [np.zeros((1, rx))]])
            ry += 1
            
            
        # ---- Algorithm step 3 ----
        # check if POD compression is needed
        r0 = self.max_rank
        if r0:
                
            if rx > r0:
                evals, evecs = np.linalg.eig(self.Gx)
                idx = np.argsort(evals)
                idx = idx[-1:-1-r0:-1]   # indices of largest r0 eigenvalues
                qx = evecs[:, idx]
                self.Qx = self.Qx.dot(qx)
                self.A = self.A.dot(qx)
                self.Gx = np.diag(evals[idx])
                
            if ry > r0:
                evals, evecs = np.linalg.eig(self.Gy)
                idx = np.argsort(evals)
                idx = idx[-1:-1-r0:-1]   # indices of largest r0 eigenvalues
                qy = evecs[:, idx]
                self.Qy = self.Qy.dot(qy)
                self.A = qy.T.dot(self.A)
                self.Gy = np.diag(evals[idx])

        # ---- Algorithm step 4 ----
        xtilde = self.Qx.T.dot(x)
        ytilde = self.Qy.T.dot(y)
        # assert np.allclose(xtilde,xtilde)
        # assert np.allclose(ytilde,ytilde)
        # update A and Gx

        self.A += ytilde.dot(xtilde.T)
        self.Gx += xtilde.dot(xtilde.T)
        self.Gy += ytilde.dot(ytilde.T)
        
        # assert np.allclose(self.A, self.A)
        # assert np.allclose(self.Gx, self.Gx)
        # assert np.allclose(self.Gy, self.Gy)
        
    def compute_matrix(self):
        return self.Qx.T.dot(self.Qy).dot(self.A).dot(np.linalg.pinv(self.Gx))

    def _eigendecomp(self):
        Ktilde = self.compute_matrix()
        self._eigs, self._evecK = np.linalg.eig(Ktilde)
        return self

    # def _sdmd_modes(self):
    #     print('modes')
    #     return self.Qx.dot(self._evecK)
    # def compute_modes(self):
    #     Ktilde = self.compute_matrix()
    #     evals, evecK = np.linalg.eig(Ktilde)
    #     # try:    
    #         modes = self.Qx.dot(evecK)
    #     # except MemoryError:
    #     #     try:
    #     #         modes = np.zeros_like(self.Qx, dtype=np.complex128)
    #     #         for i in range(evecK.shape[0]):
    #     #             modes += np.outer(self.Qx[:, i].ravel(), evecK[i, :].ravel())
    #     #     except MemoryError:
    #     #         p = comm_tools.find_dir('MA_code_dev')
    #     #         path = os.path.join(p, '../test_streaming')
    #             np.savez(os.path.join(path, 'compute_modes_ele'), Qx=self.Qx, evecK=evecK, evals=evals)
    #             modes=None
    #     return modes, evals
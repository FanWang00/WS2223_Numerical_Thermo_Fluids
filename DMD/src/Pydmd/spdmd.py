import numpy as np
# from scipy.linalg import pinv2
# --> Import PyDMD base class for DMD.
from .dmdbase import DMDBase
# from .sparse import SparseDMD

class SPDMD(DMDBase):
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
        super(SPDMD, self).__init__(svd_rank=0, tlsq_rank=0, exact=False, 
                                    opt=True, nx=sparse_dmd.nx, 
                                    ny=sparse_dmd.ny, n_plot=10)
        self.dmd = sparse_dmd.dmd
        self.sparse_dmd = sparse_dmd.sparse

        self.nmodes = self.sparse_dmd.Nz[number_index]
        self.Ni = number_index

        self.data_shape = shape
        self.axis = axis

        self.rmodes = self.sparse_reconstruction()

        nonzero = self.sparse_dmd.nonzero[:, number_index]

        self.modes = self.dmd.modes[:, nonzero]
        self.freqs = self.dmd.eigs[nonzero]
        self.amplitudes = self.sparse_dmd.xpol[nonzero, number_index]
        self.ploss = self.sparse_dmd.Ploss[number_index]
        
    @property
    def reconstructed_data(self):
        return sparse_reconstruction()
    
    def reconstructed_data_save(self, out_mode='save_reshape'):
        if not out_mode in ['std_reshape', 'save_reshape', 'no_reshape']:
            raise ValueError("Wrong save mode, out mode should be 'std_reshape', 'save_reshape' or 'no_reshape'")
        if out_mode == 'std_reshape': 
            return np.moveaxis(np.reshape(self.reconstructed_data, (self.nx, self.ny, -1)), [0, 1], [1, 0])
        if out_mode == 'save_reshape': 
            return np.reshape(reconstructed_data, (self.nx, self.ny, -1)).T
        if out_mode == 'no_reshape': 
            return sparse_reconstruction()
    
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



class SPDMD_recons(DMDBase):
    def __init__(self, sparse_dmd, number_index,
                 svd_rank=0, tlsq_rank=0, exact=False, opt=False, nx=None, ny=None,
                 n_sig=10):
    # print(exact)
        super(SPDMD_recons, self).__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank, exact=exact, 
                                    opt=opt, nx=sparse_dmd.dmd.nx, ny=sparse_dmd.dmd.ny, n_sig=n_sig)
        self.sp_dmd = None
        self.Ni = number_index
        self.dmd = sparse_dmd.dmd
        self.sparse_dmd = sparse_dmd.sparse
        self.nmodes = self.sparse_dmd.Nz[number_index]
        self.ploss = self.sparse_dmd.Ploss[number_index]
        # self.rmodes = self.sparse_reconstruction()

        # self.modes = self.dmd.modes[:, nonzero]
        # self.freqs = self.dmd.Edmd[nonzero]
        # self.amplitudes = self.sparse_dmd.xpol[nonzero, number_index]
        self.ploss = self.sparse_dmd.Ploss[number_index]
        self.nonzero = self.sparse_dmd.nonzero[:, number_index]
    @property
    def eigs(self):
        return self.sparse_dmd.dmd.eigs
    
    @property
    def mode_idx(self):
        # a = np.where(self.nonzero, 1, 0)
        a = self.nonzero
        b = np.asarray([i for i in range(len(a))])
        # print(b)
        # print(a)
        return b[a]
     
    @property
    def dynamics(self):
        return self.dmd.sp_vander[self.nonzero]
    
    @property
    def modes(self):
        return self.dmd.modes[:, self.nonzero]
    
    @property
    def amplitudes(self):
        """
        Get the reduced Koopman operator A, called A tilde.

        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.sparse_dmd.xpol[self.nonzero, self.Ni]

    @property
    def reconstructed_data(self): 
        return np.dot(self.modes, np.dot(np.diag(self.amplitudes), self.dynamics))
    
    # def fit(self, dmd, gammaval, MP=False):
    #     self.sp_dmd = SparseDMD(dmd)
    #     self.sp_dmd.compute_sparse(gammaval, MP=MP)
    
    # def get_Ni(self, Ni):
    #     """
    #      Ni : number idx for nonzeros
    #     """
    #     self.Ni = Ni
    #     self.nonzero = self.sparse_dmd.nonzero[:, Ni]
    # def sp_recons(self, Ni):
    #     return self.sp_dmd.reconstruction(Ni)
        
    
        

    
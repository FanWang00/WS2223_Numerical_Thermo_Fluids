"""
Base module for the DMD: `fit` method must be implemented in inherited classes
"""
# from __future__ import division

import sys
# sys.path.append("..")
import warnings
from builtins import object
from builtins import range
from os.path import splitext
# from Pydmd. import DMDBase
# from utils_dmd.decorator_np_test import np_agnostic 
import matplotlib as mpl
import os 
import numpy as np 
# from past.utils import old_div
import dmd_tools
from dmd_tools import dmd_cri16
from dmd_cmap import colormap_dmd, r_Spectral
from multiprocessing import Pool
from functools import partial
import comm_tools
import matplotlib.colors as colors
# import help_func
# fro import dmd_cri16
from matplotlib.colors import LinearSegmentedColormap
# try:
#     import cupy as cp
# except Exception:
#     cp = None

# global np
# np = np

mpl.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt
# try:
#     import cupy as cp
# except:
#     import 
# def colormap_dmd():
#     cmap = np.load('../CCool.npy')
#     n_bins = cmap.shape[0]
#     name_cmap = 'DMD'
#     cm = LinearSegmentedColormap.from_list(name=name_cmap, colors=cmap, N=n_bins)
#     return cm


class DMDBase(object):
    """
    Dynamic Mode Decomposition base class.

    :param int svd_rank: rank truncation in SVD. If 0, the method computes the
        optimal rank and uses it for truncation; if positive number, the method
        uses the argument for the truncation; if -1, the method does not
        compute truncation.
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimized DMD. Default is False.
    :cvar dict original_time: dictionary that contains information about the
        time window where the system is sampled:

           - `t0` is the time of the first input snapshot;
           - `tend` is the time of the last input snapshot;
           - `dt` is the delta time between the snapshots.

    :cvar dict dmd_time: dictionary that contains information about the time
        window where the system is reconstructed:

            - `t0` is the time of the first approximated solution;
            - `tend` is the time of the last approximated solution;
            - `dt` is the delta time between the approximated solutions.

    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False, nx=None, ny=None,
                    n_sig=10):
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.opt = opt
        self.original_time = None
        self.dmd_time = None
        self.original_time_input = None
        self.dmd_time_input = None
        self.nx = nx
        self.ny = ny
        self._eigs = None
        self._Atilde = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None
        self.n_sig=n_sig
        self.P = None
        self.q = None
        # super().__init__(svd_rank=svd_rank, tlsq_rank=tlsq_rank, exact=exact, opt=opt)
        # self.np = None

    # @property
    # def cmap_dmd():
        
    #     cm = colormap_dmd()
    #     return cm

    @property
    def dmd_timesteps(self):
        """
        Get the timesteps of the reconstructed states.

        :return: the time intervals of the original snapshots.   
        :rtype: numpy.ndarray
        """
        # print(self.dmd_time_input)
        if self.dmd_time_input is not None:
            return self.dmd_time_input 
        else:
            return np.arange(self.dmd_time['t0'],
                            self.dmd_time['tend'] + self.dmd_time['dt'],
                            self.dmd_time['dt'])

    @property
    def original_timesteps(self):
        """
        Get the timesteps of the original snapshot.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        if self.original_time_input is not None:
            return self.original_time_input

        #     self.original_time['dt'] = self.original_time_input[1]-original_time_input[0]
        #     self.original_time['dt'] = self.original_time_input[0]
        #     self.original_time['tend'] = self.original_time_input[-1]
            # return self.original_time_input
        else:
            return np.arange(self.original_time['t0'],
                            self.original_time['tend'] + self.original_time['dt'],
                            self.original_time['dt'])

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.
        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        return self._modes


    def modes_save(self, out_mode='save_reshape'):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        if out_mode not in ['std_reshape', 'save_reshape', 'no_reshape']:
            raise ValueError("Wrong save mode, out mode should be 'std_reshape', 'save_reshape' or 'no_reshape'")
        if out_mode == 'std_reshape': 
            return np.moveaxis(np.reshape(self._modes, (self.nx, self.ny, -1)), [0, 1], [1, 0])
        if out_mode == 'save_reshape': 
            return np.reshape(self._modes, (self.nx, self.ny, -1)).T
        if out_mode == 'no_reshape': 
            return self._modes
        # switcher = {
        #             'std_reshape': np.moveaxis(np.reshape(self._modes, (self.nx, self.ny, -1)), [0, 1], [1, 0]),
        #             'save_reshape': np.reshape(self._modes, (self.nx, self.ny, -1)).T ,
        #             'no_reshape': self._modes
        #             }
        # print(self.nx, self.ny)
        # print(np.moveaxis(np.reshape(self._modes, (self.nx, self.ny, -1)), [0, 1], [1, 0]))
        # return switcher[out_mode]

    @property
    def atilde(self):
        """
        Get the reduced Koopman operator A, called A tilde.

        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self._Atilde

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return self._eigs

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray
        """
        # omega = old_div(np.log(self.eigs), self.original_time['dt'])
        omega = np.log(self.eigs) / self.original_time['dt']
        # print(self.dmd_timesteps)
        vander = np.exp(np.multiply(*np.meshgrid(omega, self.dmd_timesteps)))
        return (vander * self.amplitudes).T
    

        
    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        print(f'shape of dynamics {self.dynamics.shape}')
        return self.modes.dot(self.dynamics).real

    def partial_reconstructed_data(self, idx, out_mode='save_reshape'):
        """
        Get reconstructed data by idx list
        """
        if not isinstance(idx, (list, np.ndarray)):
            idx = [idx]
        print(f'idx: {idx}')
        # print(idx)
        if not out_mode in ['std_reshape', 'save_reshape', 'no_reshape']:
            raise ValueError("Wrong save mode, out mode should be 'std_reshape', 'save_reshape' or 'no_reshape'")
        if out_mode == 'std_reshape': 
            return np.moveaxis(np.reshape(self.modes[:, idx].dot(self.dynamics[idx, :]).real 
            (self.nx, self.ny, -1)), [0, 1], [1, 0])
        if out_mode == 'save_reshape': 
            return np.reshape(self.modes[:, idx].dot(self.dynamics[idx, :]).real, (self.nx, self.ny, -1)).T
        if out_mode == 'no_reshape': 
            return np.real(self.modes[:, idx].dot(self.dynamics[idx, :]))
  

    # @property
    def reconstructed_data_save(self, out_mode='save_reshape'):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        # return np.reshape(self.modes.dot(self.dynamics), (self.nx, self.ny, -1)).T
        if not out_mode in ['std_reshape', 'save_reshape', 'no_reshape']:
            raise ValueError("Wrong save mode, out mode should be 'std_reshape', 'save_reshape' or 'no_reshape'")
        if out_mode == 'std_reshape': 
            return np.moveaxis(np.reshape(self._modes.dot(self.dynamics).real, (self.nx, self.ny, -1)), [0, 1], [1, 0])
        if out_mode == 'save_reshape': 
            return np.reshape(self._modes.dot(self.dynamics).real, (self.nx, self.ny, -1)).T
        if out_mode == 'no_reshape': 
            return self._modes.dot(self.dynamics).real
    @property
    def snapshots(self):
        """
        Get the original input data.

        :return: the matrix that contains the original snapshots.
        :rtype: numpy.ndarray
        """
        return self._snapshots

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """
        return np.log(self.eigs).imag / (2 * np.pi * self.original_time['dt'])
    
    @staticmethod
    def freq_su(D, U):
        # for Re =100
        # dt = 0.2
        # T = 300 * dt
        Su = 0.16
        f = Su * U / D
        return f
        


    @property
    def amplitudes(self):
        """
        Get the coefficients that minimize the error between the original
        system and the reconstructed one. For futher information, see
        `dmdbase._compute_amplitudes`.

        :return: the array that contains the amplitudes coefficient.
        :rtype: numpy.ndarray
        """
        return self._b
    
    @property
    def _power(self):
        return np.array([np.linalg.norm(self.modes[:, i], 2) ** 2 for i in
                        range(self.modes.shape[1])])
    
    def get_dmd_time(self, time):
        self.dmd_time_input=time
    
    def get_ori_time(self, time):
        self.dmd_time_input = time.copy()
        self.original_time = {'t0': time[0], 'tend': time[-1], 'dt': time[1]-time[0]}
        self.original_time_input = time

    def _sort(self, sort_mode='amp'):
        """
        sort
        """
        # print(sort_mode)
        # f = self.frequency
        print(f"sort_method: {sort_mode}")
        if sort_mode not in ['frequencies', 'power', 'amp', 'cri16', 'eigmag', None]: raise ValueError
        # assert(len(self.frequency) == len(self._power))

        if sort_mode == 'amp':
            return  np.argsort(np.absolute(self.amplitudes))[::-1]
        elif sort_mode == 'cri16':
            return dmd_cri16(self.dynamics, self.modes)[::-1]
        elif sort_mode == 'eigmag':
            return np.argsort(np.abs(self.eigs))[::-1]
        elif sort_mode == 'frequencies':
            return np.argsort(self.frequency)[::-1]   
        else:
            return np.asarray([i for i in range(self._snapshots.shape[1]-1)])
            


    # @property
    def item_sort_idx(self, m='amp'):
        return self._sort(sort_mode=m)
        
    # @property
    # def item_sort(self, m='amp'):
    #     switcher = {
    #         'amp': np.absolute(self.amplitudes)[self._sort(m)],
    #         'power': np.sort(self._power),
    #         'modes': abs(np.sort(-np.linalg.norm(self._modes, ord=1, axis=0))),        
    #         }
    #     return switcher[m]
        
        # return abs(np.sort(-np.linalg.norm(self._b, axis=0)))
    
    @property
    def snapshots_shape(self):
        return self.snapshots.shape
    
    @property
    def r_svd(self):

        return self.eigs.shape[0]
    
    @property
    def lowrank_svd_data(self):
        """
        Get the lowrank svd data.

        :return: the tuple that contains lowrank svd data .
        :rtype: tuple
        """
        return self._U_lowrank, self._s_lowrank, self._V_lowrank
    
    
    def fit(self, X):
        """
        Abstract method to fit the snapshots matrices.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.fit'.format(
                self.__class__.__name__))

    @staticmethod
    def _col_major_2darray(X):
        """
        Private method that takes as input the snapshots and stores them into a
        2D matrix, by column. If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.

        :param X: the input snapshots.
        :type X: int or numpy.ndarray
        :return: the 2D matrix that contains the flatten snapshots, the shape
            of original snapshots.
        :rtype: numpy.ndarray, tuple
        """
        
        # If the data is already 2D ndarray
        if isinstance(X, np.ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError('Snapshots have not the same dimension.')

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.asarray(x).flatten() for x in X])

        # check condition number of the data passed in
        cond_number = np.linalg.cond(snapshots)
        if cond_number > 10e4:
            print("Input data matrix X has condition number {}. "
                "Consider preprocessing data, passing in augmented data matrix, or regularization methods."
                .format(cond_number))
            # warnings.warn("Input data matrix X has condition number {}. "
            #               "Consider preprocessing data, passing in augmented data matrix, or regularization methods."
            #               .format(cond_number))

        return snapshots, snapshots_shape

    
    @staticmethod
    def _compute_tlsq(X, Y, tlsq_rank):
        """
        Compute Total Least Square.

        :param numpy.ndarray X: the first matrix;
        :param numpy.ndarray Y: the second matrix;
        :param int tlsq_rank: the rank for the truncation; If 0, the method
            does not compute any noise reduction; if positive number, the
            method uses the argument for the SVD truncation used in the TLSQ
            method.
        :return: the denoised matrix X, the denoised matrix Y
        :rtype: numpy.ndarray, numpy.ndarray

        References:
        https://arxiv.org/pdf/1703.11004.pdf
        https://arxiv.org/pdf/1502.03854.pdf
        """
        # Do not perform tlsq
        if tlsq_rank == 0:
            return X, Y

        V = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)[-1]
        rank = min(tlsq_rank, V.shape[0])
        VV = V[:rank, :].conj().T.dot(V[:rank, :])

        return X.dot(VV), Y.dot(VV)

    @staticmethod
    def _compute_svd(X, svd_rank):
        """
        Truncated Singular Value Decomposition.

        :param numpy.ndarray X: the matrix to decompose.
        :param svd_rank: the rank for the truncation; If 0, the method computes
            the optimal rank and uses it for truncation; if positive interger,
            the method uses the argument for the truncation; if float between 0
            and 1, the rank is the number of the biggest singular values that
            are needed to reach the 'energy' specified by `svd_rank`; if -1,
            the method does not compute truncation.
        :type svd_rank: int or float
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        # try:
        #     np = cp.get_array_module(X)
        # except Exception:
        #     np = np

        U, s, V = np.linalg.svd(X, full_matrices=False)
        V = V.conj().T

        if svd_rank == 0:
            omega = lambda x: 0.56 * x ** 3 - 0.95 * x ** 2 + 1.82 * x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif svd_rank > 0 and svd_rank < 1:
            cumulative_energy = np.cumsum(s ** 2 / (s ** 2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = X.shape[1]

        U = U[:, :rank]
        V = V[:, :rank]
        s = s[:rank]
        return U, s, V

    @staticmethod
    # @np_agnostic
    def _build_lowrank_op(U, s, V, Y):
        """
        Private method that computes the lowrank operator from the singular
        value decomposition of matrix X and the matrix Y.

        .. math::

            \\mathbf{\\tilde{A}} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{X}^\\dagger \\mathbf{U} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{V} \\mathbf{S}^{-1}

        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param numpy.ndarray Y: input matrix Y.
        :return: the lowrank operator
        :rtype: numpy.ndarray
        """
        return U.T.conj().dot(Y).dot(V) * np.reciprocal(s)

    @staticmethod
    # @np_agnostic
    def _eig_from_lowrank_op(Atilde, Y, U, s, V, exact):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator from the low-dimensional operator and the
        input matrix.

        :param numpy.ndarray Atilde: the lowrank operator.
        :param numpy.ndarray Y: input matrix Y.
        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param bool exact: if True, the exact modes are computed; otherwise,
            the projected ones are computed.
        :return: eigenvalues, eigenvectors
        :rtype: numpy.ndarray, numpy.ndarray
        """
        lowrank_eigenvalues, lowrank_eigenvectors = np.linalg.eig(Atilde)

        # Compute the eigenvectors of the high-dimensional operator
        if exact:
            eigenvectors = (
                (Y.dot(V) * np.reciprocal(s)).dot(lowrank_eigenvectors))
        else:
            eigenvectors = U.dot(lowrank_eigenvectors)

        # The eigenvalues are the same
        eigenvalues = lowrank_eigenvalues

        return eigenvalues, eigenvectors


    # @np_agnostic
    def _compute_amplitudes(self, modes, snapshots, eigs, opt):
        """
        Compute the amplitude coefficients. If `opt` is False the amplitudes
        are computed by minimizing the error between the modes and the first
        snapshot; if `opt` is True the amplitudes are computed by minimizing
        the error between the modes and all the snapshots, at the enpense of
        bigger computational cost.

        :param numpy.ndarray modes: 2D matrix that contains the modes, stored
            by column.
        :param numpy.ndarray snapshots: 2D matrix that contains the original
            snapshots, stored by column.
        :param numpy.ndarray eigs: array that contains the eigenvalues of the
            linear operator.
        :param bool opt: flag for computing the optimal amplitudes of the DMD
            modes, minimizing the error between the time evolution and all
            the original snapshots. If false the amplitudes are computed
            using only the initial condition, that is snapshots[0].
        :return: the amplitudes array
        :rtype: numpy.ndarray

        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        """
        
        if opt:
            # compute the vandermonde matrix
            # print(eigs)
            # print(self.original_time)
            # omega = old_div(np.log(eigs), self.original_time['dt'])
            omega = np.log(eigs) / self.original_time['dt']
            vander = np.exp(
                np.multiply(*np.meshgrid(omega, self.dmd_timesteps))).T

            # perform svd on all the snapshots
            U, s, V = np.linalg.svd(self._snapshots, full_matrices=False)

            self.P = np.multiply(
                np.dot(modes.conj().T, modes),
                np.conj(np.dot(vander, vander.conj().T)))
            tmp = (np.dot(np.dot(U, np.diag(s)), V)).conj().T
            self.q = np.conj(np.diag(np.dot(np.dot(vander, tmp), modes)))

            # b optimal
            a = np.linalg.solve(self.P, self.q)
        else:
            a = np.linalg.lstsq(modes, snapshots.T[0], rcond=None)[0]

        return a

    def plot_eigs(self, show_eig_num=False,
                  sort_mode='cri16',
                  show_axes=True,
                  n_sig=None,
                  n_plot=None,
                  show_unit_circle=True,
                  figsize=(8, 8),
                  title=''):
        """
        Plot the eigenvalues.

        :param bool show_axes: if True, the axes will be showed in the plot.
            Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
            and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        :param str title: title of the plot.
        """ 
        if n_sig is None:
            n_sig = self.n_sig

        if self._eigs is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the fit method.')
        if n_plot == None:
            n_plot = len(self.eigs)
            # idx, _ = self.amplitudes(sort=True)
        # col = np.where(x<1,'k',np.where(y<5,'b','r'))
        # print(f"{n_sig=}")
        # print(f"{sort_mode=}")
        marker_size, col, num_marker = self._plot_setup(sort_mode, n_sig)

        # col = ['r']* len(self._eigs)
        # marker_szie = [10] * len(self._eigs)
        # if len(self._eigs) >=10:
        #     marker_szie[:10] = [40] * 10
        #     col[10:] = ['b'] * (len(self._eigs) -10)
        
        # num_marker = [i for i in range(len(self._eigs))]

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()
        # print(col)
        points = ax.scatter(
        # self._eigs.real, self._eigs.imag, 'bo', label='Eigenvalues')
        self._eigs.real[:n_plot], self._eigs.imag[:n_plot], c=col[:n_plot], marker='o' , label='Eigenvalues', s=marker_size)

        # set limits for axis
        limit = np.max(np.ceil(np.absolute(self._eigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        plt.ylabel('Imaginary part')
        plt.xlabel('Real part')
        
        if show_eig_num:
            markder_idx =  self.item_sort_idx(m=sort_mode)[:n_sig]
            for x, y, num in zip(self._eigs.real[markder_idx], self._eigs.imag[markder_idx], num_marker):
                plt.text(x, y, num, color="red", fontsize=12)
            # ax.add_artist(eig_num)
    

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0., 0.),
                1.,
                color='green',
                fill=False,
                label='Unit circle',
                linestyle='--')
            ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-.')
        ax.grid(True)

        ax.set_aspect('equal')

        # x and y axes
        if show_axes:
            ax.annotate(
                '',
                xy=(np.max([limit * 0.8, 1.]), 0.),
                xytext=(np.min([-limit * 0.8, -1.]), 0.),
                arrowprops=dict(arrowstyle="->"))
            ax.annotate(
                '',
                xy=(0., np.max([limit * 0.8, 1.])),
                xytext=(0., np.min([-limit * 0.8, -1.])),
                arrowprops=dict(arrowstyle="->"))

        # legend
        if show_unit_circle:
            ax.add_artist(
                plt.legend(
                    [points, unit_circle], ['Eigenvalues', 'Unit circle'],
                    loc=1))
        else:
            ax.add_artist(plt.legend([points], ['Eigenvalues'], loc=1))

        plt.show()

    def plot_modes_2D(self,
                      index_mode=None,
                      filename=None,
                      x=None,
                      y=None,
                      order='C',
                      figsize=(8, 8)):
        """
        Plot the DMD Modes.

        :param index_mode: the index of the modes to plot. By default, all
            the modes are plotted.
        :type index_mode: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param order: read the elements of snapshots using this index order,
            and place the elements into the reshaped array using this index
            order.  It has to be the same used to store the snapshot. 'C' means
            to read/ write the elements using C-like index order, with the last
            axis index changing fastest, back to the first axis index changing
            slowest.  'F' means to read / write the elements using Fortran-like
            index order, with the first index changing fastest, and the last
            index changing slowest.  Note that the 'C' and 'F' options take no
            account of the memory layout of the underlying array, and only
            refer to the order of indexing.  'A' means to read / write the
            elements in Fortran-like index order if a is Fortran contiguous in
            memory, C-like order otherwise.
        :type order: {'C', 'F', 'A'}, default 'C'.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self._modes is None:
            raise ValueError('The modes have not been computed.'
                             'You have to perform the fit method.')

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    'No information about the original shape of the snapshots.')

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    'The dimension of the input snapshots is not 2D.')

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_mode is None:
            index_mode = list(range(self._modes.shape[1]))
        elif isinstance(index_mode, int):
            index_mode = [index_mode]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_mode:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('DMD Mode {}'.format(idx))

            real_ax = fig.add_subplot(1, 2, 1)
            imag_ax = fig.add_subplot(1, 2, 2)

            mode = self._modes.T[idx].reshape(xgrid.shape, order=order)

            real = real_ax.pcolormesh(
                xgrid,
                ygrid,
                mode.real,
                cmap='jet',
                vmin=mode.real.min(),
                vmax=mode.real.max())
            imag = imag_ax.pcolormesh(
                xgrid,
                ygrid,
                mode.imag,
                vmin=mode.imag.min(),
                vmax=mode.imag.max())

            fig.colorbar(real, ax=real_ax)
            fig.colorbar(imag, ax=imag_ax)

            real_ax.set_aspect('auto')
            imag_ax.set_aspect('auto')

            real_ax.set_title('Real')
            imag_ax.set_title('Imag')

            # padding between elements
            plt.tight_layout(pad=2.)

            if filename:
                plt.savefig('{0}.{1}{2}'.format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()

    def plot_snapshots_2D(self,
                          index_snap=None,
                          filename=None,
                          x=None,
                          y=None,
                          order='C',
                          figsize=(8, 8)):
        """
        Plot the snapshots.

        :param index_snap: the index of the snapshots to plot. By default, all
            the snapshots are plotted.
        :type index_snap: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param order: read the elements of snapshots using this index order,
            and place the elements into the reshaped array using this index
            order.  It has to be the same used to store the snapshot. 'C' means
            to read/ write the elements using C-like index order, with the last
            axis index changing fastest, back to the first axis index changing
            slowest.  'F' means to read / write the elements using Fortran-like
            index order, with the first index changing fastest, and the last
            index changing slowest.  Note that the 'C' and 'F' options take no
            account of the memory layout of the underlying array, and only
            refer to the order of indexing.  'A' means to read / write the
            elements in Fortran-like index order if a is Fortran contiguous in
            memory, C-like order otherwise.
        :type order: {'C', 'F', 'A'}, default 'C'.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self._snapshots is None:
            raise ValueError('Input snapshots not found.')

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    'No information about the original shape of the snapshots.')

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    'The dimension of the input snapshots is not 2D.')

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_snap is None:
            index_snap = list(range(self._snapshots.shape[1]))
        elif isinstance(index_snap, int):
            index_snap = [index_snap]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_snap:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('Snapshot {}'.format(idx))

            snapshot = (self._snapshots.T[idx].real.reshape(
                xgrid.shape, order=order))

            contour = plt.pcolormesh(
                xgrid,
                ygrid,
                snapshot,
                vmin=snapshot.min(),
                vmax=snapshot.max())

            fig.colorbar(contour)

            if filename:
                plt.savefig('{0}.{1}{2}'.format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()

    def plot_recons_error(self, snaps=None, num=0, idx=None):
        
        snap_size = self.nx * self.ny
        if idx is not None:
            recons_data = self.partial_reconstructed_data(idx, out_mode='no_reshape')[num*snap_size:(num+1)*snap_size]
            # recons_data = self.partial_reconstructed_data[num*snap_size:(num+1)*snap_size]
        # recons_data = self.reconstructed_data_save('no_reshape')
        else:
            recons_data = self.reconstructed_data[num*snap_size:(num+1)*snap_size]
        # print(recons_data.shape)
        if snaps is None:
            snaps = self.snapshots[num*snap_size:(num+1)*snap_size]
        fig, axs = plt.subplots(figsize=(16, 10))
        axs.plot([i for i in range(recons_data.shape[-1])], 
                    np.linalg.norm(snaps - recons_data, axis=0)/np.linalg.norm(snaps, axis=0),
                    marker='v', label="L2 norm")
        axs.plot([i for i in range(recons_data.shape[-1])], 
                    np.linalg.norm(snaps - recons_data, axis=0, ord=1)/np.linalg.norm(snaps, axis=0, ord=1),
                    marker='o', label="L1 norm")
        axs.set_ylabel(r'$\dfrac{\left\Vert X-X_{recons}  \right\Vert_p}{\left\Vert X \right \Vert_p}$', 
                        fontsize=20)
        axs.set_xlabel('Time step', fontsize=20)
        plt.legend(loc='upper right')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.yscale('log')
        plt.title('reconstruction error')
        return axs
        
    
    def _plot_setup(self, sort_mode='cri16', n_sig=None):
        
        if n_sig is None:
            n_sig = self.n_sig
        # print(n_sig)
        N_tot = len(self.eigs)
        col = np.array(['b']* N_tot)
        marker_size = [10] * N_tot
        marker_size = np.array(marker_size)
        if N_tot >=n_sig:
            marker_size[self.item_sort_idx(m=sort_mode)[:n_sig]] = [40] * n_sig
            col[self.item_sort_idx(m=sort_mode)[:n_sig]] = ['r'] * (n_sig)
        else:
            col = np.array(['b']* N_tot)
        # if N_tot >=n_sig:
        #     marker_size[self.item_sort_idx(m=sort_mode)][:n_sig] = [40] * n_sig
        #     col[self.item_sort_idx(m=sort_mode)][:n_sig] = ['r'] * (n_sig)
        # else:
        #     col = np.array(['b']* N_tot)

        num_marker = [i for i in range(n_sig)]
        return marker_size, col, num_marker
    
    def plot_grid(self, plot_idx, sort_mode='cri16', nx=None, ny=None, cmap=colormap_dmd()):
        
        if self.nx is not None:
            nx = self.nx
        elif nx is None: 
            raise ValueError('Pleas input value of nx...')
        else:
            self.nx = nx
        
        if self.ny is not None:
            ny = self.ny
        elif ny is None:
            raise ValueError('Pleas input value of ny...')
        else:
            self.ny = ny
        
        n_item = 5
        n_modes = len(plot_idx)
        cols_tag = ['Real', 'Img', 'Dynamics', 'Magnitude','Phase']
        rows_tag = [f"mode {i}" for i in range(n_modes)]
        amp_idx = self.item_sort_idx(m=sort_mode)
        plot_idx = amp_idx[plot_idx]
        fig, axs = plt.subplots(n_modes, n_item, figsize=(n_item*4, n_modes*2),
                                subplot_kw={'xticks': [], 'yticks': []})
        
        _modes = self.modes
        # vmin = [None]*5
        # vmax = [None]*5
        norm_p = [None]*5
        # vmin[0] = _modes.real.min()
        # vmax[0] = _modes.real.max()
        # vmin[1] = _modes.imag.min()
        # vmax[1] = _modes.imag.max()

        
        phase_angle = np.angle(_modes)
        # vmin[4] = phase_angle.min()
        # vmax[4] = phase_angle.max()
        # vmin.append(self.modes.)
        
        # fig = plt.figure(constrained_layout=True, figsize=(20, 10))
        # gs = gridspec.GridSpec(n_modes, n_item, figure=fig)
        # axs = [[None]*n_item]*n_modes
        for i, p in enumerate(plot_idx):
            # for j in range(n_item):
            tt = self.dmd_timesteps
            # axs[i, 0] = fig.add_subplot(gs[i, 0])
            value_amp_tmp = _modes[:, p]
            value_tmp = np.reshape(value_amp_tmp, (nx, ny,)).T

            v_0 = value_tmp.real
            # v_0f = np.linalg.norm(v_0)
            # if v_0f == 0:
            #     v_tmp_real = np.zeros_like(v_0)
            # else:
            #     # v_tmp_real = np.divide(v_0, v_0f, out=np.zeros_like(v_0), where=v_0f!=0)
            #     v_tmp_real = v_0 / v_0f

            maxmin_0 = np.abs(v_0 ).max()
            p_norm_0 = colors.Normalize(vmin=-maxmin_0, vmax=maxmin_0)
            axs[i, 0].pcolormesh(v_0 , norm=p_norm_0, cmap=cmap)
            # c_bar1 = fig.colorbar(im, ax=ax, shrink=1)
            axs[i, 0].set_yticklabels([])
            axs[i, 0].set_xticklabels([])
            axs[i, 0].label_outer()
            
            # axs[i, 1] = fig.add_subplot(gs[i, 1])
            # v_1 = value_tmp.imag/np.linalg.norm(value_tmp.imag)
            # v_tmp_imag[v_tmp_imag is np.nan] = 0
            v_1 = value_tmp.imag
            v_1f = np.linalg.norm(v_1)
            # v_tmp_imag = np.divide(v_1, v_1f, out=np.zeros_like(v_1), where=v_1f!=0)
            if v_1f == 0:
                v_tmp_imag = np.zeros_like(v_1)
            else:
                # v_tmp_real = np.divide(v_0, v_0f, out=np.zeros_like(v_0), where=v_0f!=0)
                v_tmp_imag = v_1 / v_1f
            # maxmin_1 = np.abs(value_tmp.imag).max()
            maxmin_1 = np.abs(v_tmp_imag).max()
            if v_1f == 0:
                maxmin_1 = 0.001
            # print(v_tmp_imag)
            p_norm_1 = colors.Normalize(vmin=-maxmin_1, vmax=maxmin_1)
            # p_norm_1=None
            axs[i, 1].pcolormesh(v_tmp_imag, norm=p_norm_1, cmap=cmap)
            axs[i, 1].set_yticklabels([])
            axs[i, 1].set_xticklabels([])
            
            # axs[i, 2] = fig.add_subplot(gs[i, 2])
            axs[i, 2].plot(tt, self.dynamics[p].real, label='Re')
            axs[i, 2].plot(tt, self.dynamics[p].imag, label='Imag')
            dy_min, dy_max = axs[i, 2].get_ylim()
            # print(dy_min, dy_max)
            axs[i, 2].set_yticks([dy_min, dy_max])
            axs[i, 2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

            axs[i, 3].pcolormesh(abs(value_tmp), cmap='jet')
            axs[i, 3].set_yticklabels([])
            axs[i, 3].set_xticklabels([])
            
            phase_angle_tmp = np.reshape(phase_angle[:, p], (nx, ny)).T
            axs[i, 4].pcolormesh(phase_angle_tmp, cmap='plasma')
            axs[i, 4].set_yticklabels([])
            axs[i, 4].set_xticklabels([])
        for ax, col in zip(axs[0], cols_tag):
            ax.set_title(col)

        for ax, row in zip(axs[:,0], rows_tag):
            ax.set_ylabel(row, size='large')
        

    def plot_freq(self, n_sig=None, sort_mode='cri16', figsize=(8, 8), title=''):
        # max_amp = am
        marker_size, col, num_marker  = self._plot_setup(sort_mode=sort_mode, n_sig=n_sig)

        y_max = np.absolute(self.amplitudes).max()
        y = np.absolute(self.amplitudes)/y_max
        if sort_mode == 'amp':
            y_label = 'Normalized amplitude'
        if sort_mode == 'cri16':
            y_label = 'Normalized amplitude(sort by criterion16)'

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()
        # plt.scatter(abs(dmd.frequency), amp_magnitude/amp_max, 
        #         c=col, marker='o', s=marker_szie)

        markerline, stemlines, _  = ax.stem(abs(self.frequency), y, use_line_collection=True)
        line = ax.get_lines()
        print(line[0].get_xdata().shape)
        # print(np.stack(stemlines.get_segments()).shape)
        xd = line[0].get_xdata()
        yd = line[0].get_ydata()
        n_segms = len(stemlines.get_segments())
        # n_segms = len(stemlines)
        
        # xd = [line[i].get_xdata() for i in range(n_segms)]
        # yd = [line[i].get_ydata() for i in range(n_segms)]
        
        for xx, yy, ms, cc in zip(xd, yd, marker_size/4, col):
            plt.plot(xx, yy, 'o', ms=ms, mfc=cc, mec=cc)
            # plt.setp(sl, 'color', 'r')
        
        markder_idx =  self.item_sort_idx(m=sort_mode)[:n_sig]
        tx = [markerline.get_xdata()[i] for i in markder_idx]
        ty = [markerline.get_ydata()[i]*1.4 for i in markder_idx]
        for x, y, num in zip(tx, ty, num_marker):
            plt.text(x, y, num, color="red", fontsize=12, wrap=True)

        plt.yscale('log')
        plt.ylabel(y_label)
        plt.xlabel('frequency(Hz)')
        plt.show()
        
     

    def plot_growthrate(self, n_sig=None, sort_mode='amp', figsize=(8, 8), title=''):

        marker_size, col, num_marker  = self._plot_setup(sort_mode=sort_mode, n_sig=n_sig)
        if sort_mode == 'amp':
            y_label = 'Normalized amplitude'
        if sort_mode == 'cri16':
            y_label = 'Normalized amplitude(sort by criterion16)'

        y_max = np.absolute(self.amplitudes).max()
        y_magnitude = np.absolute(self.amplitudes)
        

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()
        # plt.scatter(abs(dmd.frequency), amp_magnitude/amp_max, 
        #         c=col, marker='o', s=marker_szie)
        markerline, stemlines, _  = ax.stem(np.log(np.absolute(self.eigs)), y_magnitude/y_max, use_line_collection=True)
        line = ax.get_lines()
        xd = line[0].get_xdata()
        yd = line[0].get_ydata()
        # xd = [line[i].get_xdata() for i in range(len(stemlines))]
        # yd = [line[i].get_ydata() for i in range(len(stemlines))]
        
        for xx, yy, ms, cc in zip(xd, yd, marker_size/4, col):
            plt.plot(xx, yy, 'o', ms=ms, mfc=cc, mec=cc)
            # plt.setp(sl, 'color', 'r')

        # markder_idx =  self.item_sort_idx(m=sort_mode)[:n_sig]
        # tx = [markerline.get_xdata()[i]*2 for i in markder_idx]
        # ty = [markerline.get_ydata()[i] for i in markder_idx]
        # for x, y, num in zip(tx, ty, num_marker):
        #     plt.text(x, y, num, color="red", fontsize=12, wrap=True)

        plt.yscale('log')
        plt.ylabel(y_label)
        plt.xlabel('Growth Rate 1/sec)')
        plt.show()
         
    # def plot_phase(self, ncols, nrows, nx, ny, n_sig=None, figsize=(8, 8), title=''):
        
    #     # marker_size, col, num_marker  = self._plot_setup(n_sig=n_sig)
    #     amp_max = self.item_sort[0]
    #     amp_magnitude = np.absolute(self.amplitudes)
  
    #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
    #                             figsize=(ncols*4, nrows*2),
    #                             subplot_kw={'xticks': [], 'yticks': []})       
    #     value_real_amp =np.angle(dmd.modes[:, amp_idx[:nrows*ncols]])
    #     for ax, value in zip(axs.flat, np.reshape(value_real_amp, (nx, ny, -1)).T):
    #         ax.imshow(value, cmap='plasma')
    def plot_save_recons(self, save_path_dir, tp2=None, prefix=None, save_img=False, time_step=None, cm=colormap_dmd(), N_minmax=None,
                        sym_norm=False, cal_minmax=True, vmin=None, vmax=None, save_num=None, Nc=1, title=None, norm=False, nproc=1, MP=False):
        print("saving recons data...")
        ## save recons png
        if cal_minmax:
            value, _vmin, _vmax = comm_tools.save_png_assemble([self.reconstructed_data_save()], minmax=True)
            if vmin is None:
                vmin = _vmin
            if vmax is None:
                vmax = _vmax
        else:
            value = comm_tools.save_png_assemble([self.reconstructed_data_save()], minmax=False)
        

        if N_minmax is not None :
            vmin, vmax = N_minmax
            value = dmd_tools.skl_scaler(value, vmin, vmax)

        if tp2 is None:
             tp2 = ['recons data'] * self.dmd_timesteps.shape[0]
        if time_step is None:
            time_step = dmd_tools.time_step_str(self.dmd_timesteps) 
        # param_tuple = zip(value, png_save_path, time_step)
        # dmd_tools.save_png_plt_par_std(nproc=6, Nc=1, vmin=vmin, vmax=vmax, save_img=True, tp2=tp2, title='', 
        #                             param_tuple=param_tuple, cm='Spectral')
        dmd_tools.plot_save_png(value, save_path=save_path_dir, time_step=time_step, sym_norm=sym_norm,
                                tp2=tp2, Nc=Nc, vmin=vmin, vmax=vmax, save_img=save_img, title=title, cm=cm,
                                MP=MP, nproc=nproc)

    def plot_save_modes(self, save_path_dir, prefix='mode', vmin=None, vmax=None, Nc=1, sym_norm=True, tp2=None, save_img=True, cm=colormap_dmd(),
                        sort_mode=None, save_num=None, title=None, nproc=2, MP=False):
        
        if  save_num is None or save_num >=  self.r_svd :
            save_num = self.r_svd
        if sort_mode is None:
            mode = self.modes_save()[:save_num]
        else:
            idx = self.item_sort_idx(m=sort_mode)
            # print(self.modes_save().shape)
            mode = self.modes_save()[idx[:save_num]]
            # print(mode.shape)
        for m in ['real', 'image']:
            if m == 'real':
                print('saving real modes')
                data, _, _ = comm_tools.save_png_assemble([mode.real], minmax=True)
                # data = mode.real
                # vmin = mode.min()
                # vmax = mode.max()
            if m == 'image':
                data, v_min, v_max = comm_tools.save_png_assemble([mode.imag], minmax=True)
                # data, vmin, vmax = mode.imag
                # vmin = mode.min()
                # vmax = mode.max()
                print('saving imaginary modes')
            digs = comm_tools.count_digits(self.r_svd)
            # print(save_num)
            png_save_path = [os.path.join(save_path_dir, m, f'{prefix}_{i:0{digs}}') for i in range(save_num)]
            # if sym_norm:
            #     v = np.asarray([np.absolute([i, j]).max() for i, j in zip(v_min, v_max)])
            #     print(v)
            #     vmin_p = -v
            #     vmax_p = v

            comm_tools.make_dir(os.path.dirname(png_save_path[0]))
            if tp2 is None:
                tp2 = [f'mode {i:0{digs}}' for i in range(save_num)]
            # print(tp2)
            # print(comm_tools.dim_list(data))
                # print(len(png_save_path))
                # print(png_save_path)
            # print(self.dmd_timesteps.shape)
            # time_step = [None] * save_num
            time_step = ['']*save_num
            # print(sym_norm)
            dmd_tools.plot_save_png(data=data, save_path=png_save_path, time_step=time_step, sym_norm=sym_norm,
                                    tp2=tp2, Nc=Nc, vmin=vmin, vmax=vmax, save_img=save_img, title=title,
                                    MP=MP, nproc=nproc)

            # dmd_tools.save_png_plt_std(sym_norm=sym_norm, vmin=vmin, vmax=vmax, Nc=1, save_img=save_img, tp2=tp2, title='',
            #                             data=data, save_path=png_save_path, time_step=time_step)
            # param_list = list(zip(data, png_save_path, time_step))
            # dmd_tools.save_png_plt_par_std(vmin=vmin, vmax=vmax, tp2=tp2, title='',
            #                                 save_img=save_img, nproc=nproc, 
            #                                 param_tuple=param_list)
    

    def plot_save_dynamics(self, save_path_dir, prefix='mode', W=5, H=3, sort_mode=None, save_num=None, time_step=None):
        
        print('saving dynamics')
        if  save_num is None or save_num >=  self.r_svd :
            save_num = self.r_svd
        if sort_mode is None:
            idx = [i for i in range(save_num)]
            dy_data = self.dynamics
        else:
            idx = self.item_sort_idx(m=sort_mode)[:save_num]
            dy_data = self.dynamics[idx]
        if time_step is None:
            time_step = self.dmd_timesteps
        # png_save_path = [os.path.join(save_path_dir, f'{prefix}_{i}') for i in range(save_num)]
        png_save_path = [os.path.join(save_path_dir, f'{prefix}_{i}') for i in iter(idx)]
        dmd_tools.plot_save_dynamics(save_path_dir, dy_data=dy_data, prefix=prefix, 
                                    save_num=save_num, time_step=time_step, W=W, H=H)
        
    def plot_save_mode_sep(self, save_path_dir, prefix=None, save_img=False, time_step=None, sort_mode=None,
                            sym_norm=False, vmin=None, vmax=None, tp2_suffix='', 
                            save_num=None, idx_SepMode=None, 
                            Nc=1, title=None, 
                            norm=False, cm=colormap_dmd(), nproc=1, MP=False):
        print('saving recons by separating modes')
        if idx_SepMode is None:
            if  save_num is None or save_num >=  self.r_svd:
                save_num = self.r_svd
            if sort_mode is None:
                # dy_data = self.modes_save()
                idx = [i for i in range(save_num)]
            else:
                idx = self.item_sort_idx(m=sort_mode)[:save_num]
            
            idx_SepMode = zip([i for i in range(len(idx))], idx)
            # dy_data = self.dynamics[idx]
        digs = comm_tools.count_digits(self.r_svd)
        
        for i, k in idx_SepMode:
            print(f'save No.{i} mode with index {k} separately')
            print('entering assemble')
            data, vmin, vmax = comm_tools.save_png_assemble([self.partial_reconstructed_data(k)], minmax=True, 
                                clean=True, clean_threshold=0.1)
            t =  len(data)
            digs_t = comm_tools.count_digits(t)
            if not norm:
                vmin = None
                vmax = None
            if prefix is None:
                png_save_path = [os.path.join(save_path_dir, f"mode_{k:0{digs}}", f"recons_{j:0{digs_t}}")
                                for j in range(t)]
            else:
                png_save_path = [os.path.join(save_path_dir, f"mode_{k:0{digs}}", f"{prefix[j]}")for j in range(t)]
            comm_tools.make_dir(os.path.dirname(png_save_path[0]))
            
            tp2 = [f'mode {k:0{digs}} ({tp2_suffix})'] * t
            # print(len(data))
            # print(len(png_save_path))
            # print(png_save_path)
            # print(self.dmd_timesteps.shape)
            if time_step is None:
                time_step = dmd_tools.time_step_str(self.dmd_timesteps)
            # param_list = list(zip(data, png_save_path, time_step))

            # param_list = list(zip(data, png_save_path, self.dmd_timesteps))
            
            dmd_tools.plot_save_png(data=data, save_path=png_save_path, time_step=time_step, sym_norm=sym_norm,
                                    tp2=tp2, Nc=Nc, vmin=vmin, vmax=vmax, save_img=save_img, title=title, cm=cm,
                                    MP=MP, nproc=nproc)
        return png_save_path
    
    def plot_dynamics(self, Nr, Nc, sort_mode=None, plot_mode='grid', save_img=False):
        '''
        plot_mode: 'single mode': plot single plot
                   'grid mode': plot grid plot with many pics
        '''  
        if plot_mode == 'grid':
            idx = self.item_sort_idx(m=sort_mode)
            tt = self.dmd_timesteps
            dynamics_real = self.dynamics[idx[:Nr*Nc], :].real
            dynamics_imag = self.dynamics[idx[:Nr*Nc], :].imag
            
            # fig, axs = plt.subplots(Nr, Nc, figsize=(Nr*6, Nc*4))
            fig, axs = plt.subplots(nrows=Nr, ncols=Nc, figsize=(Nc*4, Nr*3),
                                    # subplot_kw={'xticks': [], 'yticks': []},
                                    # sharex='col',
                                    # sharey='row',
                                    # gridspec_kw={'hspace': 0, 'wspace': 0},
                                    # gridspec_kw={'wspace': 0}
                                    )
            axs = axs.flatten()
            # axs = [axs]
            handles = []
            labels = []
            for i in range(min(self.dynamics.shape[0], Nc*Nr)):
                # plt.xticks([])
                # plt.yticks([])
                # plot_idx = amp_idx[i]
                # print(plot_idx)
                # pos = axs[0].imshow(value_real, cmap=cm)

                # for ax, value in zip(axs, [dynamics_real, dynamics_imag]):
                axs[i].plot(tt, dynamics_real[i], label='Re')
                axs[i].plot(tt, dynamics_imag[i], label='Imag')
                axs[i].set_title(f"dynamics of mode {i}")
            axs[0].legend(loc='upper right')
            #     ax.legend()    #     print(value)
                # axs[1].imshow(value_imag, cmap=cm)
                # axs[0].set_title(f"real part of dyanmics mode {plot_idx:03d}")
                # axs[0].set_title(f"imaginary part dynamics of mode {plot_idx:03d}")
                # plt.contour(value, levels=1, colors='k', linewidths=1.2)
                # # plt.colorbar(pos)
                # fig.subplots_adjust(right=0.8)
                # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                # fig.colorbar(im, cax=cbar_ax, extend='max')
            #     handle_tmp, label_tmp = ax.get_legend_handles_labels()
            #     handles.append(handle_tmp)
            #     labels.append(label_tmp)
            # handles, labels = axs[-1].get_legend_handles_labels()
            # axs[0].legend(handles, labels, bbox_to_anchor=(0, 1.35), loc='upper left')
            # Create the legend
            # fig.legend([l1, l2, l3, l4],     # The line objects
            #            labels=line_labels,   # The labels for each line
            #            loc="center right",   # Position of legend
            #            borderaxespad=0.1,    # Small spacing around legend box
            #            title="Legend Title"  # Title for the legend
            #            )
            fig.suptitle('dynamics of modes sorted by amplitudes', fontsize=30)
            # plt.tight_layout()
            plt.show()


    def mode_dynamics_video(self, savepath, n_plot=2, sort_mode=None, save_img=False, nproc=1, cmap=colormap_dmd()):
        '''
        param: idx: first n dominate modes and dynamics  
        '''
        idx = self.item_sort_idx(m=sort_mode)[:n_plot]
        time_step = self.dmd_timesteps
        # step = [i for i in range(time_step.shape[0])]
        recons = np.stack([self.partial_reconstructed_data(i) for i in idx], axis=1)
        
        if nproc > 1:
            # def parallel(time_step=time_setp, nx=self.nx, ny=self.ny, savepath=savepath, save_img=save_img):
            #         return partial(dmd_tools.mode_dynamics_video, nx, ny, savepath, save_img)
            par = partial(cmap, dmd_tools.mode_dynamics_video, nx, ny, save_img, modes, dynamics, MP)
            pool = Pool(nproc)
            input_args = list(zip(time_step, savepath))
            pool.starmap(par, input_args)
            pool.terminate()
            pool.join
        else:
            dmd_tools.mode_dynamics_video(nx=self.nx, ny=self.ny, save_img=save_img, 
                                          modes=self.modes[:, idx], dynamics=self.dynamics[idx, :], 
                                          MP=False, cmap=cmap,
                                          time_step=time_step, savepath=savepath, recons=recons)
    
    def compare_mode_cri(self, N=10):
        
        L1_err = []
        L2_err = [] 
        L1_snap = np.linalg.norm(self._snapshots, axis=0, ord=1)
        L2_snap = np.linalg.norm(self._snapshots, axis=0, ord=2)
        amp_idx = self.item_sort_idx(m='amp')[:N]
        cri16_idx = self.item_sort_idx(m='cri16')[:N]
        eigmag_idx  = self.item_sort_idx(m='eigmag')[:N]

        sort_method = ['amp', 'cri16', 'eigmag']
        
        x = [i for i in range(self._snapshots.shape[-1])]
        fig, axs = plt.subplots(figsize=(16, 10))
        
        for idx in [amp_idx, cri16_idx, eigmag_idx]:
            
            recons_data = self._modes[:, idx].dot(self.dynamics[idx])
            L1_err_tmp = np.linalg.norm((self._snapshots - recons_data), ord=1, axis=0)/L1_snap
            L1_err.append(L1_err_tmp)
            # print(L1_err_tmp.shape)
            # L1_err.append(L1_err_tmp)
            L2_err_tmp = np.linalg.norm((self._snapshots - recons_data), ord=2, axis=0)/L2_snap
            L2_err.append(L2_err_tmp)

        colors = plt.cm.jet(np.linspace(0, 1, len(L1_err)))
        for i in range(len(L1_err)):
            
            print(f"selected mode idx of {sort_method[i]}")
            # L2_err.append(L2_err_tmp)
            # # print(self.r_svd)
            # print(i)
            axs.plot(x, L1_err[i], color=colors[i], marker='o', label=f"L1 norm with {sort_method[i]}", markevery=20)
            axs.plot(x, L2_err[i], color=colors[i], marker='x', label=f"L2 norm with {sort_method[i]}", markevery=20)
            
        # for idx in [amp_idx, cri16_idx]:
        axs.set_ylabel(r'$\frac{\left\Vert X-X_{recons}  \right\Vert_p}{\left\Vert X \right \Vert_p}$', 
                        fontsize=30)
        axs.set_xlabel('Time step', fontsize=30)
        handles, labels = axs.get_legend_handles_labels()
        handles = np.reshape(handles, (-1, 2))
        labels = np.reshape(labels, (-1, 2))
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('reconstruction error')
        plt.show()
    
    def save_img(self):
        
        dmd_tools.save_png_plt_std(cm=colormap_dmd(), vmin=None, vmax=None, Nr=1, save_img=False, tp2=None, title='',
                 data=None, save_path=None, time_step=None)
        
    # def avi_out(self, title)

def plot_EigVal(eigs=None,
          show_axes=True,
          show_unit_circle=True,
          show_eig_num = True,
          figsize=(8, 8),
          title=''):
    """
    Plot the eigenvalues.

    :param bool show_axes: if True, the axes will be showed in the plot.
        Default is True.
    :param bool show_unit_circle: if True, the circle with unitary radius
        and center in the origin will be showed. Default is True.
    :param tuple(int,int) figsize: tuple in inches defining the figure
        size. Default is (8, 8).
    :param str title: title of the plot.
    """
    if eigs is None:
        raise ValueError('The eigenvalues have not been computed.'
                         'You have to perform the fit method.')
    # idx, _ = self.amplitudes(sort=True)
    # col = np.where(x<1,'k',np.where(y<5,'b','r'))
    
    col = ['r']* len(eigs)
    marker_szie = [10] * len(eigs)
    if len(eigs) >=10:
        marker_szie[:10] = [40] * 10
        col[10:] = ['b'] * (len(eigs) -10)
    
    num_marker = [i for i in range(len(eigs))]

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.gcf()
    ax = plt.gca()

    points = ax.scatter(
        # self._eigs.real, self._eigs.imag, 'bo', label='Eigenvalues')
        eigs.real, eigs.imag, c=col, marker='o' , label='Eigenvalues', s=marker_szie)

    # set limits for axis
    limit = np.max(np.ceil(np.absolute(eigs)))
    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))

    plt.ylabel('Imaginary part')
    plt.xlabel('Real part')
    
    if show_eig_num:
        for x, y, num in zip(eigs.real, eigs.imag, num_marker):
            plt.text(x, y, num, color="red", fontsize=12)
        # ax.add_artist(eig_num)
    
    if show_unit_circle:
        unit_circle = plt.Circle(
            (0., 0.),
            1.,
            color='green',
            fill=False,
            label='Unit circle',
            linestyle='--')
        ax.add_artist(unit_circle)

    # Dashed grid
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')
    ax.grid(True)

    ax.set_aspect('equal')

    # x and y axes
    if show_axes:
        ax.annotate(
            '',
            xy=(np.max([limit * 0.8, 1.]), 0.),
            xytext=(np.min([-limit * 0.8, -1.]), 0.),
            arrowprops=dict(arrowstyle="->"))
        ax.annotate(
            '',
            xy=(0., np.max([limit * 0.8, 1.])),
            xytext=(0., np.min([-limit * 0.8, -1.])),
            arrowprops=dict(arrowstyle="->"))

    # legend
    if show_unit_circle:
        ax.add_artist(
            plt.legend(
                [points, unit_circle], ['Eigenvalues', 'Unit circle'],
                loc=1))
    else:
        ax.add_artist(plt.legend([points], ['Eigenvalues'], loc=1))

    plt.show()
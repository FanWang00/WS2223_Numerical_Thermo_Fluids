
"""
Derived module from dmdbase.py for multi-resolution dmd.
Reference:
- Kutz, J. Nathan, Xing Fu, and Steven L. Brunton. Multiresolution Dynamic Mode
Decomposition. SIAM Journal on Applied Dynamical Systems 15.2 (2016): 713-735.
"""

from __future__ import division
import sys
# sys.path.append("..")
from builtins import range
from past.utils import old_div
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
# from help_func import plot_recons_error
from .dmdbase import DMDBase
from dmd_tools import colormap_dmd, dmd_cri16
import warnings
warnings.filterwarnings("error")
class MrDMD(DMDBase):
    """
    Multi-resolution Dynamic Mode Decomposition
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    :param int max_cycles: the maximum number of mode oscillations in any given
        time scale. Default is 1.
    :param int max_level: the maximum number of levels. Defualt is 6.
    """

    def __init__(self,
                 svd_rank=0,
                 tlsq_rank=0,
                 exact=False,
                 nx = None,
                 ny = None,
                 opt=False,
                 max_cycles=1,
                 max_level=6):
        super(MrDMD, self).__init__(svd_rank, tlsq_rank, exact, opt, nx, ny)
        self.max_cycles = max_cycles
        self.max_level = max_level
        self._nsamples = None
        self._steps = None

    def _index_list(self, level, node):
        """
        Private method that return the right index element from a given level
        and node.
        :param int level: the level in the binary tree.
        :param int node: the node id.
        :rtype: int
        :return: the index of the list that contains the binary tree.
        """
        if level >= self.max_level:
            raise ValueError("Invalid level: greater than `max_level`")

        if node >= 2**level:
            raise ValueError("Invalid node")

        return 2**level + node - 1

    def _index_list_reversed(self, index):
        """
        Method that return the level and node given the index of the bin.
        :param int index: the index of the bin in the binary tree.
        :return: the level of the bin in the binary tree and the node id
            in that level.
        """
        if index > 2**self.max_level - 2:
            raise ValueError("Invalid index: maximum index is ({})".format(2**self.max_level - 2))
        for lvl in range(self.max_level + 1):
            if index < 2**lvl - 1:
                break
        level = lvl - 1
        node = index - 2**level + 1
        return level, node

    @property
    def _count_modes(self):
        """
        count modes, dynamics of by level and nodes
        """
        # node = type('count', (object,), {})()
        c = []
        for i in self._b:
            c.append(len(i))
        return c
    
    def _stitch(self, level):
        """
        stitch modes, dynamics from bins by level
        """
        if level >= self.max_level:
            raise ValueError(f"level is larger than max level={self.max_level}")
        for lvl in len(level):
            pass
            



    def partial_time_interval(self, level, node):
        """
        Evaluate the start and end time and the period of a given bin.
        :param int level: the level in the binary tree.
        :param int node: the node id.
        :return: the start and end time and the period of the bin
        :rtype: dictionary
        """
        if level >= self.max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less than the '
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self.max_level))

        if node >= 2**level:
            raise ValueError("Invalid node")

        full_period = self.original_time['tend'] - self.original_time['t0']
        period = full_period / 2**level
        t0 = self.original_time['t0'] + period*node
        tend = t0 + period
        return {'t0': t0, 'tend':tend, 'dt':period}

    def time_window_bins(self, t0, tend):
        """
        Find which bins are embedded (partially or totally) in a given
        time window.
        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: indexes of the bins seen by the time window.
        :rtype: numpy.ndarray
        """
        indexes = []
        for level in range(self.max_level):
            for i in range(2**level):
                local_times = self.partial_time_interval(level, i)
                if t0 >= local_times['t0'] and t0 < local_times['tend']:
                    indexes.append(self._index_list(level, i))
                if tend > local_times['t0'] and tend <= local_times['tend']:
                    indexes.append(self._index_list(level, i))
                if t0 <= local_times['t0'] and tend >= local_times['tend']:
                    indexes.append(self._index_list(level, i))
        # Remove duplicates if they exist
        # indexes = list(dict.fromkeys(indexes)) # Python 3.7 or later (preserve order)
        indexes = list(set(indexes)) # Any Python version, but does not preserve order
        indexes = np.sort(indexes)
        return indexes

    def time_window_eigs(self, t0, tend):
        """
        Get the eigenvalues relative to the modes of the bins embedded (partially
        or totally) in a given time window.
        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the eigenvalues for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        return np.concatenate([self._eigs[idx] for idx in indexes])

    def time_window_frequency(self, t0, tend):
        """
        Get the frequencies relative to the modes of the bins embedded (partially
        or totally) in a given time window.
        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the frequencies for that time window.
        :rtype: numpy.ndarray
        """
        eigs = self.time_window_eigs(t0, tend)
        return np.log(eigs).imag/(2*np.pi*self.original_time['dt'])

    def time_window_growth_rate(self, t0, tend):
        """
        Get the growth rate values relative to the modes of the bins embedded (partially
        or totally) in a given time window.
        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the Floquet values for that time window.
        :rtype: numpy.ndarray
        """
        return self.time_window_eigs(t0, tend).real/self.original_time['dt']

    def time_window_amplitudes(self, t0, tend):
        """
        Get the amplitudes relative to the modes of the bins embedded (partially
        or totally) in a given time window.
        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the amplitude of the modes for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        return np.concatenate([self._b[idx] for idx in indexes])

    # @property
    # def reconstructed_data(self):
    #     """
    #     Get the reconstructed data.
    #     :return: the matrix that contains the reconstructed snapshots.
    #     :rtype: numpy.ndarray
    #     """
    #     try:
    #         data = np.sum(
    #             np.array([
    #                 self.partial_reconstructed_data(i)
    #                 for i in range(self.max_level)
    #             ]),
    #             axis=0)
    #     except MemoryError:
    #         data = np.array(self.partial_reconstructed_data(0))
    #         for i in range(1, self.max_level):
    #             data = np.sum([data,
    #             np.array(self.partial_reconstructed_data(i))], axis=0)
    #     return data
    # def _sort(self, sort_mode='amp'):
    #     """Sort frequencies and power according to one or the other
    #     according to `sort`

    #     Parameters
    #     --------
    #     f : Array-like
    #         List of frequencies
    #     P : Array-like
    #         List of powers
    #     sort : String
    #         Indicate to sort both frequency and power in
    #         nondecreasing order either by the frequencies given
    #         or by the power given.  Valid options: 'frequencies', 'power' 

    #     Returns
    #     --------
    #     f : Array-like
    #         Sorted frequencies
    #     P : Array-like
    #         Sorted powers
    #     """
    #     # f = self.frequency
    #     if sort_mode not in ['frequencies', 'power', 'amp', 'cri16']: raise ValueError
    #     assert(len(self.frequency) == len(self._power))
        
    #     switcher = {
    #                 'amp': np.argsort(np.absolute(self._b)),
    #                 # 'modes': np.argsort(-np.linalg.norm(self._modes, ord=1, axis=0))
    #                 'power': np.argsort(self._power),
    #                 'frequencies': np.argsort(self.frequency),
    #                 'cri16': dmd_cri16(self.dynamics, self.modes)
    #                 }
    #     return switcher[sort_mode][::-1]

    def _rank_count(self):
        pass

    
    def reconstructed_data(self, level=None):
        """
        Get the reconstructed data.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        if level is None:
            level = self.max_level
        elif level > self.max_level:
            raise ValueError("input level is larger than max level, please check!")
        
        return self._reconstructed_data(level)


    def _reconstructed_data(self, level):
        """
        private funciton of reconstructed_data
        Get the reconstructed data.
        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        try:
            data = np.sum(
                np.array([
                    self.partial_reconstructed_data(i)
                    for i in range(level)
                ]),
                axis=0)
        except MemoryError:
            data = np.array(self.partial_reconstructed_data(0))
            for i in range(1, level):
                data = np.sum([data,
                np.array(self.partial_reconstructed_data(i))], axis=0)
        return data

    
    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.
        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        return np.hstack(tuple(self._modes))

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.
        :return: the matrix that contains all the time evolution, stored by
                row.
        :rtype: numpy.ndarray
        """
        return np.vstack(
            tuple([self.partial_dynamics(i) for i in range(self.max_level)]))
    @property
    def amplitudes(self):
        return np.concatenate(self._b)

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.
        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return np.concatenate(self._eigs)

    def partial_modes(self, level, node=None):
        """
        Return the modes at the specific `level` and at the specific `node`; if
        `node` is not specified, the method returns all the modes of the given
        `level` (all the nodes).
        :param int level: the index of the level from where the modes are
            extracted.
        :param int node: the index of the node from where the modes are
            extracted; if None, the modes are extracted from all the nodes of
            the given level. Default is None.
        """
        if node:
            return self._modes[self._index_list(level, node)]

        indeces = [self._index_list(level, i) for i in range(2**level)]
        return np.hstack(tuple([self._modes[idx] for idx in indeces]))

    def partial_dynamics(self, level, node=None):
        """
        Return the time evolution of the specific `level` and of the specific
        `node`; if `node` is not specified, the method returns the time
        evolution of the given `level` (all the nodes).
        :param int level: the index of the level from where the time evolution
            is extracted.
        :param int node: the index of the node from where the time evolution is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.
        """

        def dynamic(eigs, amplitudes, step, nsamples):
            omega = old_div(
                np.log(np.power(eigs, old_div(1., step))),
                self.original_time['dt'])
            partial_timestep = np.arange(nsamples) * self.dmd_time['dt']
            try:
                vander = np.exp(np.multiply(*np.meshgrid(omega, partial_timestep)))
                dy = (vander * amplitudes).T
            except RuntimeWarning:
                print('RuntimeWarning')
                n = omega.shape[0]
                tt = partial_timestep.shape[0]
                dy = np.zeros((n, tt))
            return dy

        if node:
            indeces = [self._index_list(level, node)]
        else:
            indeces = [self._index_list(level, i) for i in range(2**level)]

        level_dynamics = [
            dynamic(self._eigs[idx], self._b[idx], self._steps[idx],
                    self._nsamples[idx]) for idx in indeces
        ]
        return scipy.linalg.block_diag(*level_dynamics)

    def partial_eigs(self, level, node=None):
        """
        Return the eigenvalues of the specific `level` and of the specific
        `node`; if `node` is not specified, the method returns the eigenvalues
        of the given `level` (all the nodes).
        :param int level: the index of the level from where the eigenvalues is
            extracted.
        :param int node: the index of the node from where the eigenvalues is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.
        """
        if level >= self.max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less than the'
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self.max_level))
        if node:
            return self._eigs[self._index_list(level, node)]

        indeces = [self._index_list(level, i) for i in range(2**level)]
        return np.concatenate([self._eigs[idx] for idx in indeces])

    def partial_reconstructed_data(self, level, node=None):
        """
        Return the reconstructed data computed using the modes and the time
        evolution at the specific `level` and at the specific `node`; if `node`
        is not specified, the method returns the reconstructed data
        of the given `level` (all the nodes).
        :param int level: the index of the level.
        :param int node: the index of the node from where the time evolution is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.
        """
        if level >= self.max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less than the '
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self.max_level))
        modes = self.partial_modes(level, node)
        dynamics = self.partial_dynamics(level, node)

        return modes.dot(dynamics).real

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.
        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        # To avoid recursion function, use FIFO list to simulate the tree
        # structure
        data_queue = [self._snapshots.copy()]

        current_bin = 0

        # Redefine max level if it is too big.
        lvl_threshold = int(np.log(self._snapshots.shape[1]/4.)/np.log(2.)) + 1
        if self.max_level > lvl_threshold:
            self.max_level = lvl_threshold
            print('Too many levels... '
                  'Redefining `max_level` to {}'.format(self.max_level))
        n_samples = self._snapshots.shape[1]          
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}

        # Reset the lists
        self._eigs = []
        self._Atilde = []
        self._modes = []
        self._b = []
        self._nsamples = []
        self._steps = []

        while data_queue:
            # print(len(data_queue))
            Xraw = data_queue.pop(0)
            # print(Xraw.shape)
            n_samples = Xraw.shape[1]
            # subsamples frequency to detect slow modes
            nyq = 8 * self.max_cycles

            step = max(1, int(np.floor(old_div(n_samples, nyq))))
            Xsub = Xraw[:, ::step]
            Xc = Xsub[:, :-1]
            Yc = Xsub[:, 1:]

            Xc, Yc = self._compute_tlsq(Xc, Yc, self.tlsq_rank)
            # print(f'Xc shape{Xc.shape}')
            U, s, V = self._compute_svd(Xc, self.svd_rank)
            # print(f"{self.svd_rank=}")

            # print(f"level={}s shape={s.shape}")

    
            Atilde = self._build_lowrank_op(U, s, V, Yc)

            eigs, modes = self._eig_from_lowrank_op(Atilde, Yc, U, s, V,
                                                    self.exact)
            rho = old_div(float(self.max_cycles), n_samples)
            slow_modes = (np.abs(
                old_div(np.log(eigs), (2. * np.pi * step)))) <= rho

            modes = modes[:, slow_modes]
            eigs = eigs[slow_modes]

            #---------------------------------------------------------------
            # DMD Amplitudes and Dynamics
            #---------------------------------------------------------------
            Vand = np.vander(
                np.power(eigs, old_div(1., step)), n_samples, True)
            b = self._compute_amplitudes(modes, Xc, eigs, self.opt)

            Psi = (Vand.T * b).T

            self._modes.append(modes)
            self._b.append(b)
            self._Atilde.append(Atilde)
            self._eigs.append(eigs)
            self._nsamples.append(n_samples)
            self._steps.append(step)


            if Xraw.dtype == 'float64':
                Xraw -= modes.dot(Psi).real
            else:
                Xraw -= modes.dot(Psi)

            if current_bin < 2**(self.max_level - 1) - 1:
                current_bin += 1
                half = int(np.ceil(old_div(Xraw.shape[1], 2)))
                data_queue.append(Xraw[:, :half])
                data_queue.append(Xraw[:, half:])
            else:
                current_bin += 1

        self.dmd_time = {'t0': 0, 'tend': self._snapshots.shape[1], 'dt': 1}
        self.original_time = self.dmd_time.copy()

        return self
    
    def plot_recons_error(self, level=None, full=True, node=None, show_num=None):
        
        # recons_data = self.reconstructed_data_save('no_reshape')
        # recons_data = self.reconstructed_data
        if show_num is None:
            show_num = self.snapshots_shape[1] 
        if full:
            _recons_data = self.reconstructed_data(level=level)
        else: 
            _recons_data = self.partial_reconstructed_data(level=level, node=node)
        
        if show_num is None:
            recons_data = _recons_data.shape
            raw_data = self.snapshots
        else:
            recons_data = _recons_data[:, :show_num]  
            raw_data = self.snapshots[:, :show_num]
        # print(recons_data.shape)
        fig, axs = plt.subplots(figsize=(16, 10))
        axs.plot([i for i in range(recons_data.shape[-1])], 
                    np.linalg.norm(raw_data - recons_data, axis=0)/np.linalg.norm(raw_data, axis=0),
                    marker='v', label="L2 norm")
        axs.plot([i for i in range(recons_data.shape[-1])], 
                    np.linalg.norm(raw_data - recons_data, axis=0, ord=1)/np.linalg.norm(raw_data, axis=0, ord=1),
                    marker='o', label="L1 norm")
        axs.set_ylabel(r'$\frac{\left\Vert X-X_{recons}  \right\Vert_p}{\left\Vert X \right \Vert_p}$', 
                        fontsize=30)
        axs.set_xlabel('Time step', fontsize=30)
        plt.legend(loc='upper right')
        plt.title('reconstruction error')
        return axs
    
    # def plot_freq(self, n_plot=None, sort_mode='cri16', figsize=(8, 8), title=''):
    #     # max_amp = amp
    #     marker_size, col, num_marker  = self._plot_setup(n_plot=n_plot, sort_mode=sort_mode)

        
    #     amp_max = self.amplitudes_sort()[0]
    #     amp_magnitude = np.absolute(self.amplitudes)

    #     plt.figure(figsize=figsize)
    #     plt.title(title)
    #     plt.gcf()
    #     ax = plt.gca()
    #     # plt.scatter(abs(dmd.frequency), amp_magnitude/amp_max, 
    #     #         c=col, marker='o', s=marker_szie)
    #     _, stemlines, _  = ax.stem(abs(self.frequency), amp_magnitude/amp_max)
    #     line = ax.get_lines()
    #     # xd = line[0].get_xdata()
    #     # yd = line[0].get_ydata()
    #     xd = [line[i].get_xdata() for i in range(len(stemlines))]
    #     yd = [line[i].get_ydata() for i in range(len(stemlines))]
        
    #     for xx, yy, ms, cc, sl in zip(xd, yd, marker_size/4, col, stemlines):
    #         plt.plot(xx, yy, 'o', ms=ms, mfc=cc, mec=cc)
    #         plt.setp(sl, 'color', 'r')

    #     plt.yscale('log')
    #     plt.ylabel('Normalized amplitude')
    #     plt.xlabel('frequency(Hz)')
    #     plt.show()
     
    # def plot_growthrate(self, n_plot=None, figsize=(8, 8), title=''):
         
    #     marker_size, col, num_marker  = self._plot_setup(n_plot=n_plot)
    #     amp_max = self.amplitudes_sort()[0]
    #     amp_magnitude = np.absolute(self.amplitudes)

    #     plt.figure(figsize=figsize)
    #     plt.title(title)
    #     plt.gcf()
    #     ax = plt.gca()
    #     # plt.scatter(abs(dmd.frequency), amp_magnitude/amp_max, 
    #     #         c=col, marker='o', s=marker_szie)
    #     _, stemlines, _  = ax.stem(np.log(np.absolute(self.eigs)), amp_magnitude/amp_max)
    #     line = ax.get_lines()
    #     # xd = line[0].get_xdata()
    #     # yd = line[0].get_ydata()
    #     xd = [line[i].get_xdata() for i in range(len(stemlines))]
    #     yd = [line[i].get_ydata() for i in range(len(stemlines))]
        
    #     for xx, yy, ms, cc, sl in zip(xd, yd, marker_size/4, col, stemlines):
    #         plt.plot(xx, yy, 'o', ms=ms, mfc=cc, mec=cc)
    #         plt.setp(sl, 'color', 'r')

    #     plt.yscale('log')
    #     plt.ylabel('Normalized amplitude')
    #     plt.xlabel('Growth Rate 1/sec)')
    #     plt.show()
         
    # def plot_phase(self, ncols, nrows, nx, ny, n_plot=None, figsize=(8, 8), title=''):
        
    #     # marker_size, col, num_marker  = self._plot_setup(n_plot=n_plot)
    #     amp_max = self.amplitudes_sort[0]
    #     amp_magnitude = np.absolute(self.amplitudes)
  
    #     fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
    #                             figsize=(ncols*4, nrows*2),
    #                             subplot_kw={'xticks': [], 'yticks': []})       
    #     value_real_amp =np.angle(dmd.modes[:, amp_idx[:nrows*ncols]])
    #     for ax, value in zip(axs.flat, np.reshape(value_real_amp, (nx, ny, -1)).T):
    #         ax.imshow(value, cmap='plasma')
    # def plot_dynamics(self, Nr, Nc):
    
    #     amp_max = self.amplitudes_sort()[0]
    #     amp_idx = self.amplitudes_sort_idx()
    #     tt = self.dmd_timesteps
    #     dynamics_real = self.dynamics[amp_idx[:Nr*Nc], :].real
    #     dynamics_imag = self.dynamics[amp_idx[:Nr*Nc], :].imag
        
    #     fig, axs = plt.subplots(Nr, Nc, figsize=(Nr*6, Nc*4))
    #     axs = axs.flatten()
    #     # axs = [axs]
    #     handles = []
    #     labels = []
    #     for i, ax in enumerate(axs):
    #         # plt.xticks([])
    #         # plt.yticks([])
    #         # plot_idx = amp_idx[i]
    #         # print(plot_idx)
    #         # pos = axs[0].imshow(value_real, cmap=cm)

    #         # for ax, value in zip(axs, [dynamics_real, dynamics_imag]):
    #         ax.plot(tt, dynamics_real[i], label='Re')
    #         ax.plot(tt, dynamics_imag[i], label='Imag')
    #         ax.set_title(f"dynamics of mode {i}")
    #     #     ax.legend()    #     print(value)
    #         # axs[1].imshow(value_imag, cmap=cm)
    #         # axs[0].set_title(f"real part of dyanmics mode {plot_idx:03d}")
    #         # axs[0].set_title(f"imaginary part dynamics of mode {plot_idx:03d}")
    #         # plt.contour(value, levels=1, colors='k', linewidths=1.2)
    #         # # plt.colorbar(pos)
    #         # fig.subplots_adjust(right=0.8)
    #         # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #         # fig.colorbar(im, cax=cbar_ax, extend='max')
    #     #     handle_tmp, label_tmp = ax.get_legend_handles_labels()
    #     #     handles.append(handle_tmp)
    #     #     labels.append(label_tmp)
    #     handles, labels = ax.get_legend_handles_labels()
    #     axs[0].legend(handles, labels, bbox_to_anchor=(0, 1.35), loc='upper left')
    #     # Create the legend
    #     # fig.legend([l1, l2, l3, l4],     # The line objects
    #     #            labels=line_labels,   # The labels for each line
    #     #            loc="center right",   # Position of legend
    #     #            borderaxespad=0.1,    # Small spacing around legend box
    #     #            title="Legend Title"  # Title for the legend
    #     #            )
    #     fig.suptitle('dynamics of modes sorted by amplitudes', fontsize=30)
    #     # plt.tight_layout()
    #     plt.show()
        
    def plot_eigs(self,
                  show_axes=True,
                  show_unit_circle=True,
                  figsize=(8, 8),
                  title='',
                  level=None,
                  node=None):
        """
        Plot the eigenvalues.
        :param bool show_axes: if True, the axes will be showed in the plot.
                Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
                and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: tuple in inches of the figure.
        :param str title: title of the plot.
        :param int level: plot only the eigenvalues of specific level.
        :param int node: plot only the eigenvalues of specific node.
        """
        if self._eigs is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the fit method.')

        if level:
            peigs = self.partial_eigs(level=level, node=node)
        else:
            peigs = self.eigs

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        if not level:
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i) for i in np.linspace(0, 1, self.max_level)]

            points = []
            for lvl in range(self.max_level):
                indeces = [self._index_list(lvl, i) for i in range(2**lvl)]
                eigs = np.concatenate([self._eigs[idx] for idx in indeces])

                points.append(
                    ax.plot(eigs.real, eigs.imag, '.', color=colors[lvl])[0])
        else:
            points = []
            points.append(
                ax.plot(peigs.real, peigs.imag, 'bo', label='Eigenvalues')[0])

        # set limits for axis
        limit = np.max(np.ceil(np.absolute(peigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        plt.ylabel('Imaginary part')
        plt.xlabel('Real part')

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0., 0.), 1., color='green', fill=False, linestyle='--')
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
        if level:
            labels = ['Eigenvalues - level {}'.format(level)]
        else:
            labels = [
                'Eigenvalues - level {}'.format(i)
                for i in range(self.max_level)
            ]

        if show_unit_circle:
            points += [unit_circle]
            labels += ['Unit circle']

        ax.add_artist(plt.legend(points, labels, loc='best'))
        plt.show()


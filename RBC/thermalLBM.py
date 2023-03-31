import numpy as np
import matplotlib.pyplot as plt
from os import path as osp
from pathlib import Path
import time
from lbm import BasicLBM

import utils

class ThermalLBM(BasicLBM):
    '''Class for thermal Lattice Boltzmann simulations.'''

    def __init__(self, Nx, Ny, nu, alpha, thermal_coeff, flow_boundary, temp_boundary, rho_perturb=0, dt=1):
        BasicLBM.__init__(self, Nx, Ny)

        # Calculate relaxation parameters
        # self.omega_nu = 1 / (3*nu + .5)
        # self.omega_alpha = 1 / (3*alpha + .5)
        self.omega_nu = dt/ (3*nu + .5)
        self.omega_alpha = dt / (3*alpha + .5)
        # Output resulting simulation parameters
        print('Simulation initialized for:')
        print('Nx = %i \nNy = %i' % (Nx, Ny))
        print('nu = %f \nalpha = %f\nomega_nu = %f \nomega_alpha = %f' %
              (nu, alpha, self.omega_nu, self.omega_alpha))
        print('thermal_coeff = %f' % (thermal_coeff))

        self.thermal_coeff = thermal_coeff
        self.flow_boundary = flow_boundary
        self.temp_boundary = temp_boundary

        # Set up macroscopic and microscopic variables
        self.feq = np.empty((9, Ny, Nx))
        self.rho = np.ones((Ny, Nx))
        self.geq = np.empty((9, Ny, Nx))
        self.theta = np.zeros((Ny, Nx))
        self.ux = np.zeros((Ny, Nx))
        self.uy = np.zeros((Ny, Nx))
        # self.u = np.zeros((Ny, Nx))
        # Slight perturbation of density
        self.rho += rho_perturb

        # Set initial populations to equilibrium
        self.update_eq(self.feq, self.rho, self.ux, self.uy)
        self.f = self.feq

        self.update_eq(self.geq, self.theta, self.ux, self.uy)
        self.g = self.geq

    def buoyancy(self):
        # Including buoyant body force using Boussinesq approximation
        f, theta, rho = self.f, self.theta, self.rho
        coeff = self.thermal_coeff

        f[2, :, :] += -1/3 * coeff * theta * rho
        f[4, :, :] += 1/3 * coeff * theta * rho
        f[[5, 6], :, :] += -1/12 * coeff * theta * rho
        f[[7, 8], :, :] += 1/12 * coeff * theta * rho

    def run(self, small_sim, frames, savepath):
        # Running the simulation for n steps
        
        data_path = osp.join(savepath, 'data')
        T_path = osp.join(savepath, 'png', 'T')
        u_path = osp.join(savepath, 'png', 'u')
        
        for p in [T_path, data_path, u_path]:
            Path(osp.join(savepath, p)).mkdir(exist_ok=True, parents=True)
        
        t1 = time.process_time() # Keep track of time for performance evaluation
        # num_digits = utils.count_digits(n)
        X, Y = np.meshgrid(np.linspace(0, self.Nx, self.Nx), np.linspace(self.Ny, 0, self.Ny))
        data_dicts = {'X_mesh':X, 'Y_mesh':Y}
        for j in range(frames):
            print(f'loop {j}/{frames}')
            for _ in range(small_sim):
                # Compute macroscopic quantities
                self.rho, self.theta = self.density(self.f), self.density(self.g)
                self.ux, self.uy = self.velocity(self.f, self.rho)
                u = np.sqrt(self.ux**2+self.uy**2)
                
                # Collision step
                self.g = self.collide(self.g, self.geq, self.omega_alpha, self.theta, self.ux, self.uy,
                                    self.temp_boundary)
                self.f = self.collide(self.f, self.feq, self.omega_nu, self.rho,
                                    self.ux, self.uy, self.flow_boundary)

                # Buoyany term addition
                self.buoyancy()
                
                # Streaming step
                self.stream(self.f)
                self.stream(self.g)
            dict_values = [self.ux, self.uy, u, self.theta]
            dict_keys = ['uy', 'ux', 'u', 'T']
            for k in  range(len(dict_keys)):
                data_dicts[dict_keys[k]] = dict_values[k]
            fname = f'LBM_RBC_{j:04}.hdf5'
            full_path = osp.join(data_path, fname)
            utils.save_hdf5(full_path, data_dicts)
            
            fig = plt.figure()
            plt.imsave(osp.join(u_path,f'LBM_RBC_{j:04}.png'),u, cmap='jet') 
            plt.imsave(osp.join(T_path,f'LBM_RBC_{j:04}.png'), self.theta,cmap='seismic') 
            plt.close()
        t2 = time.process_time()

        print(f'Calculation time for loop {fname}: {t2-t1}')
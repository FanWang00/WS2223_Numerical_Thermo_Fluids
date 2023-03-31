
# importing necessary classes and packages
from thermalLBM import ThermalLBM
from boundary import Boundary
import numpy as np
import utils
from pathlib import Path
import numba as nb


    # Defining parameters for simulation
Nx = 1000
Ny = 500
gy = -9.81  # acceleration of gravity in the y-direction
beta = .000101937 # expansion coefficient
Pr = 1
Ra = 1e5
dt = 1
# Calculating kinetic viscosity and thermal diffusivity from Rayleigh and Prandtl numbers
nu = np.sqrt(beta * abs(gy) * Ny**3 * Pr / Ra * 2)
# delta_t = np.sqrt(abs(gy)/Nx)
# delta_x = 1/(Nx-2)
# nu = np.sqrt(Pr/Ra)*delta_t/(delta_x*delta_x)
# alpha = np.sqrt(np.sqrt(1./(Pr*Ra))*delta_t/(delta_x*delta_x))
alpha = nu / Pr

# Setting up boundary object for flow
flow_boundary = Boundary()
boundary_array = np.zeros((Ny, Nx))
Th = 1
boundary_array[[0, -1], :] = Th
flow_boundary.init_bounceback(boundary_array)

# Setting up boundary object for temperature
temp_boundary = Boundary()

boundary_array = np.zeros((Ny, Nx))
boundary_array[0, :] = 1
temp_boundary.init_constant(boundary_array, -1)

boundary_array = np.zeros((Ny, Nx))
boundary_array[-1, :] = 1
temp_boundary.init_constant(boundary_array, 1)


# Initialize thermal simulation for Rayleigh-Benard convection
RayleighBenard = ThermalLBM(Nx, Ny, nu, alpha, gy*beta, flow_boundary, temp_boundary, )
frames = 1000
# Run simulation with graphical output
# RayleighBenard.run_vis(100)
save_path = f"I:/LBM_RBC/test/test_new_{Nx}Ra{Ra:.1E}_dt{dt:.1E}_Th{Th}_frames{frames}"
print(f'saved in {save_path}')
Path(save_path).mkdir(parents=True, exist_ok=True)
RayleighBenard.run(small_sim=50, frames=frames, savepath=save_path)
    

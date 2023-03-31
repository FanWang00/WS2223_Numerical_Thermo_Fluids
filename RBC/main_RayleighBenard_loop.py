
from thermalLBM import ThermalLBM
from boundary import Boundary
import numpy as np
from pathlib import Path

seeds = np.random.randint(0, np.iinfo(np.int32).max, 50)
for sd in seeds:
    # Defining parameters for simulation
    np.random.seed(sd)
    Nx = 500
    Ny = 200
    gy = -9.81  # acceleration of gravity in the y-direction
    beta = .000101937 # expansion coefficient
    Pr = 1
    Ra = 3e6
    dt = 1
    rho_perturb = np.random.random((Ny, Nx))*.1  
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
    # Th = 1+0.001*perturb_factor
    boundary_array[[0, -1], :] = 1
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
    RayleighBenard = ThermalLBM(Nx, Ny, nu, alpha, gy*beta, flow_boundary, temp_boundary, rho_perturb)
    frames = 1000
    # Run simulation with graphical output
    # RayleighBenard.run_vis(100)
    
    ### save path of simulation, change accordingly 
    save_path = f"I:/LBM_RBC/loop_rho_rand_1e-1/{Nx}Ra{Ra:.1E}_dt{dt:.1E}_SD{sd}"
    
    print(f'saved in {save_path}')
    Path(save_path).mkdir(parents=True, exist_ok=True)
    RayleighBenard.run(small_sim=50, frames=frames, savepath=save_path)
# for i in range(100):
    

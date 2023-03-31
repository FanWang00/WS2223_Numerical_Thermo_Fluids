class Boundary():
    # Boundary class for thermalLBM and LBM simulations.

    def __init__(self):
        self.boundaries = []  # List of all applied boundaries (except bounce-back)
        self.bb_array = False
        self.constant_array = False

    def init_bounceback(self, bb_array):
        # Initialize position of bounce-back cells.

        self.bb_array += (bb_array == 1)

    def init_constant(self, logic_array, constant):
        #Initialize position and value of constant condition for populations like
        # temperature or concentration.
        
        try:   # Check for existence of object variable constant_values
            self.constant_values[logic_array == 1] = constant  # Value array
        except AttributeError:
            self.constant_values = 0*logic_array
            self.constant_values[logic_array == 1] = constant

        self.constant_array += (logic_array == 1)  # Logic array indicating position

        if self.constant not in self.boundaries:
            self.boundaries.append(self.constant)

    def constant(self, f, rho, ux, uy):
        # Enforcing constant value boundary.
        # Intended for ADE-populations with constant values on walls.
        # Therefore fluid velocity is set to zero.

        
        rho[self.constant_array == 1] = self.constant_values[self.constant_array == 1]
        ux[self.constant_array == 1] = 0
        uy[self.constant_array == 1] = 0


    def enforce_boundaries(self, f, rho, ux, uy):
        # Enforce all initialized boundary conditions in the boundary list
        for boundary_method in self.boundaries:
            boundary_method(f, rho, ux, uy)

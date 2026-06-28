import numpy as np

class Quad4Element:
    """
    Isoparametric 2D Quad4 Element formulation.
    Handles shape functions, Jacobian mapping, and Nonlinear B-matrix generation.
    """
    
    @staticmethod
    def shape_functions(xi, eta):
        """ Evaluates the 4 shape functions at local coordinates (xi, eta). """
        return 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])

    @staticmethod
    def shape_function_deriv(xi, eta):
        """ Evaluates the derivatives of shape functions w.r.t xi and eta. """
        dN_dxi = 0.25 * np.array([-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi),  (1 - xi)])
        return np.vstack((dN_dxi, dN_deta))
        
    @staticmethod
    def calculate_jacobian(node_coords, dN_dlocal):
        """ 
        Calculates the Jacobian matrix and its determinant.
        """
        J = dN_dlocal @ node_coords
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        return J, detJ, invJ

    @staticmethod
    def b_matrix_nonlinear(dN_dx, F):
        """
        Constructs the nonlinear strain-displacement matrix (B_NL) for 
        Total Lagrangian formulation (large deformations).
        """
        B = np.zeros((3, 8))
        for i in range(4):
            Nx = dN_dx[0, i]
            Ny = dN_dx[1, i]
            
            # Node i, X-DOF contributions
            B[0, 2*i]   = F[0, 0] * Nx
            B[1, 2*i]   = F[0, 1] * Ny
            B[2, 2*i]   = F[0, 0] * Ny + F[0, 1] * Nx
            
            # Node i, Y-DOF contributions
            B[0, 2*i+1] = F[1, 0] * Nx
            B[1, 2*i+1] = F[1, 1] * Ny
            B[2, 2*i+1] = F[1, 0] * Ny + F[1, 1] * Nx
            
        return B
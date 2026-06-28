import numpy as np
from scipy.linalg import solve
from element_quad4 import Quad4Element

def solve_newton_raphson(nodes, elements, FreeDof, PrescribedDof, u_prescribed, u_total, material, max_iter=20, tol=1e-5):
    """
    Nonlinear finite element solver using the Newton-Raphson iterative method 
    under Displacement Control.
    """
    ndof = len(nodes) * 2
    
    # Update total displacements with the prescribed step
    u_total[PrescribedDof] = u_prescribed
    
    # Gauss Integration Points (2x2)
    pts = [-1.0/np.sqrt(3), 1.0/np.sqrt(3)]
    wts = [1.0, 1.0]
    
    for iteration in range(max_iter):
        K_global = np.zeros((ndof, ndof))
        F_int = np.zeros(ndof)
        
        for el in elements:
            el_nodes = nodes[el]
            el_dofs = np.array([[2*n, 2*n+1] for n in el]).flatten()
            u_el = u_total[el_dofs]
            
            K_el = np.zeros((8, 8))
            f_el = np.zeros(8)
            
            for xi, wx in zip(pts, wts):
                for eta, wy in zip(pts, wts):
                    dN_dlocal = Quad4Element.shape_function_deriv(xi, eta)
                    J, detJ, invJ = Quad4Element.calculate_jacobian(el_nodes, dN_dlocal)
                    dN_dx = invJ @ dN_dlocal
                    
                    # Compute Deformation Gradient (F = I + grad_u)
                    grad_u = np.zeros((2, 2))
                    for i in range(4):
                        grad_u[0, 0] += dN_dx[0, i] * u_el[2*i]
                        grad_u[0, 1] += dN_dx[1, i] * u_el[2*i]
                        grad_u[1, 0] += dN_dx[0, i] * u_el[2*i+1]
                        grad_u[1, 1] += dN_dx[1, i] * u_el[2*i+1]
                        
                    F_def = np.eye(2) + grad_u
                    
                    # Material evaluation
                    S, D = material.evaluate(F_def)
                    
                    # Nonlinear B-matrix assembly
                    B_NL = Quad4Element.b_matrix_nonlinear(dN_dx, F_def)
                    
                    dV = detJ * wx * wy
                    
                    # Internal forces and material stiffness matrix
                    f_el += (B_NL.T @ S) * dV
                    K_el += (B_NL.T @ D @ B_NL) * dV
                    
            # Assembly
            for i, I in enumerate(el_dofs):
                F_int[I] += f_el[i]
                for j, J_dof in enumerate(el_dofs):
                    K_global[I, J_dof] += K_el[i, j]
                    
        # Residual Calculation
        Residual = -F_int[FreeDof]
        res_norm = np.linalg.norm(Residual)
        
        print(f"    Iter {iteration+1:02d}: Residual Norm = {res_norm:.2e}")
        
        if res_norm < tol:
            break
            
        # Solve for incremental displacement
        K_ff = K_global[np.ix_(FreeDof, FreeDof)]
        delta_u = solve(K_ff, Residual)
        
        u_total[FreeDof] += delta_u
        
    # Extract Reaction Forces
    Reactions = F_int[PrescribedDof]
    return u_total, Reactions

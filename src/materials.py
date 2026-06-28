import numpy as np

class NeoHookeanMaterial:
    """
    Neo-Hookean Hyperelastic Material Model for 2D plane strain formulation.
    Computes the 2nd Piola-Kirchhoff stress tensor (S) and Material Stiffness (D).
    """
    def __init__(self, mu, bulk_K):
        self.mu = mu
        self.bulk_K = bulk_K
        
    def evaluate(self, F):
        """
        Evaluates the material response given the Deformation Gradient (F).
        Returns the 2nd Piola-Kirchhoff stress (S) and Tangent Modulus (D).
        """
        C = F.T @ F
        J = max(np.linalg.det(F), 1e-7) # Guard against zero volume
        
        # 1st Invariant (Plane strain implies C33 = 1)
        I1 = C[0, 0] + C[1, 1] + 1.0 
        
        C_inv = np.linalg.inv(C)
        c11, c22, c12 = C_inv[0, 0], C_inv[1, 1], C_inv[0, 1]
        
        # --- 2nd Piola-Kirchhoff Stress (S) ---
        c_mu = self.mu * (J ** (-2/3))
        
        # Isochoric part
        S11_iso = c_mu * (1.0 - (1.0/3.0) * I1 * c11)
        S22_iso = c_mu * (1.0 - (1.0/3.0) * I1 * c22)
        S12_iso = c_mu * (0.0 - (1.0/3.0) * I1 * c12)
        
        # Volumetric part
        p = self.bulk_K * (J - 1.0)
        S11_vol = J * p * c11
        S22_vol = J * p * c22
        S12_vol = J * p * c12
        
        S = np.array([S11_iso + S11_vol, 
                      S22_iso + S22_vol, 
                      S12_iso + S12_vol])
        
        # --- Constitutive Matrix (D) ---
        # Approximated Secant Material Stiffness for Newton-Raphson convergence
        D = np.zeros((3, 3))
        p_tilde = p + self.bulk_K * J
        
        D[0, 0] = c_mu * (4.0/9.0 * I1 * c11**2 + 2.0/3.0 * c11) + J * p_tilde * c11**2 - 2.0 * J * p * c11**2
        D[1, 1] = c_mu * (4.0/9.0 * I1 * c22**2 + 2.0/3.0 * c22) + J * p_tilde * c22**2 - 2.0 * J * p * c22**2
        D[0, 1] = c_mu * (4.0/9.0 * I1 * c11 * c22 - 2.0/3.0 * (c11+c22)) + J * p_tilde * c11 * c22 - 2.0 * J * p * c12**2
        D[1, 0] = D[0, 1]
        D[2, 2] = c_mu * (1.0/9.0 * I1 * c12**2 + 0.5 * (c11*c22 + c12**2)) + J * p_tilde * c12**2 - J * p * (c11*c22 + c12**2)
        
        D[0, 2] = c_mu * (4.0/9.0 * I1 * c11 * c12 - 2.0/3.0 * c12) + J * p_tilde * c11 * c12 - 2.0 * J * p * c11 * c12
        D[2, 0] = D[0, 2]
        D[1, 2] = c_mu * (4.0/9.0 * I1 * c22 * c12 - 2.0/3.0 * c12) + J * p_tilde * c22 * c12 - 2.0 * J * p * c22 * c12
        D[2, 1] = D[1, 2]
        
        return S, D

import numpy as np
import matplotlib.pyplot as plt
from materials import NeoHookeanMaterial
from solver_nr import solve_newton_raphson

def main():
    print("==================================================")
    print("   Nonlinear 2D FEA Solver (Total Lagrangian)     ")
    print("==================================================\n")
    
    # 1. Mesh Definition (Simple Cantilever - 2 Quad4 Elements)
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]
    ])
    
    elements = [
        [0, 1, 4, 3],
        [1, 2, 5, 4]
    ]
    ndof = len(nodes) * 2
    
    # 2. Material Properties (Neo-Hookean)
    mu = 80.194e6     # Shear modulus (Pa)
    bulk_K = 164.2e6  # Bulk modulus (Pa)
    material = NeoHookeanMaterial(mu, bulk_K)
    
    # 3. Boundary Conditions
    FixedNodes = [0, 3] # Fixed at the left wall
    PulledNodes = [2, 5] # Pulled from the right wall
    
    FixedDof = [2*n for n in FixedNodes] + [2*n+1 for n in FixedNodes] # Fix X and Y
    PrescribedDof_Pull = [2*n for n in PulledNodes] # Pull in X direction
    
    PrescribedDof = FixedDof + PrescribedDof_Pull
    FreeDof = [i for i in range(ndof) if i not in PrescribedDof]
    
    # 4. Load Stepping (Displacement Control)
    num_steps = 10
    max_disp = 1.5 # 1.5 meters of extension (Massive nonlinear deformation)
    
    displacements = []
    reactions = []
    
    u_current_prescribed = np.zeros(len(PrescribedDof))
    u_total = np.zeros(ndof)
    
    print("Starting Incremental Analysis...")
    for step in range(1, num_steps + 1):
        disp_step = max_disp * (step / num_steps)
        print(f"\n[Load Step {step}/{num_steps}] Target Displacement: {disp_step:.3f} m")
        
        # Apply displacement exclusively to pulled nodes
        for i, dof in enumerate(PrescribedDof):
            if dof in PrescribedDof_Pull:
                u_current_prescribed[i] = disp_step
        
        # Call Newton-Raphson Solver
        u_total, R = solve_newton_raphson(nodes, elements, FreeDof, PrescribedDof, 
                                          u_current_prescribed, u_total, material)
        
        # Aggregate Reaction Forces
        total_force = sum([R[i] for i, dof in enumerate(PrescribedDof) if dof in PrescribedDof_Pull])
                
        displacements.append(disp_step)
        reactions.append(total_force)
        
    # 5. Result Visualization
    print("\nSimulation Complete. Plotting validation curves...")
    
    plt.figure(figsize=(9, 6))
    plt.plot(displacements, np.array(reactions) / 1e6, color='#d62728', marker='o', 
             linewidth=2.5, markersize=8, label='Neo-Hookean Reaction')
    
    plt.title('Nonlinear Force-Displacement Curve', fontsize=15, fontweight='bold')
    plt.xlabel('Prescribed Displacement $u_x$ (m)', fontsize=13)
    plt.ylabel('Total Reaction Force (MN)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    # Save the plot for the GitHub README
    plt.savefig('force_disp_demo.png', dpi=300)
    print("Plot successfully saved as 'force_disp_demo.png'.")

if __name__ == "__main__":
    main()
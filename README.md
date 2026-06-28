# Python Nonlinear FEA Solver (2D)

A custom, object-oriented Finite Element Analysis (FEA) solver written entirely in Python from scratch. This project is capable of handling **Geometric Nonlinearity** (Large Displacements) and **Material Nonlinearity** (Hyperelasticity), validated against commercial software (ANSYS).

## 📌 Project Overview
While many engineers treat FEA software as a "black box", the development of accurate Digital Twins requires a deep understanding of the underlying computational mechanics. This solver was built to explicitly model complex nonlinear behaviors in 2D continuous structures.

The solver features **Isoparametric Quad4 elements**, a **Total Lagrangian** formulation for large deformations, and a robust **Newton-Raphson** iterative scheme for nonlinear equilibrium finding.

## 🚀 Key Implementations

* **Geometric Nonlinearity:** Handles large displacement kinematics using a Total Lagrangian approach, properly updating the deformation gradient and geometric stiffness matrices.
* **Hyperelastic Material Laws:** Beyond linear elasticity, the solver incorporates a **Neo-Hookean** material model. It analytically computes the 2nd Piola-Kirchhoff stress tensor and the tangent constitutive matrix.
* **Nonlinear Solver:** A custom implementation of the Newton-Raphson method with load-stepping and dynamic residual tolerance tracking.
* **Dynamic Analysis (Linear):** Includes a Newmark-beta integration scheme for transient dynamic response analysis.

## 📊 Validation
The solver's accuracy (displacements, principal stresses, and reaction forces) was benchmarked and strictly validated against equivalent nonlinear models built in **ANSYS**, demonstrating near-zero error margins.

*(Place an image from your `images/` folder here showing the Force-Displacement curve or stress fields)*
`![Nonlinear Response](images/force_disp.png)`

## 🛠️ Tech Stack
* **Language:** Python
* **Mathematics & Matrices:** NumPy, SciPy (`scipy.linalg` for sparse matrix inversion)
* **Visualization:** Matplotlib (for stress fields and reaction curves)

## 💻 Repository Structure
* `src/element_quad4.py`: Isoparametric shape functions, Jacobian, and B-matrix formulation.
* `src/materials.py`: Linear Elastic and Hyperelastic (Neo-Hookean) material class definitions.
* `src/solver_nr.py`: The Newton-Raphson nonlinear equilibrium solver algorithm.
* `src/main.py`: Model assembly, constraint application, and execution.

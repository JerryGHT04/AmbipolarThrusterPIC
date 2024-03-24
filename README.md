2024.3.8
This version should correctly capture the Neumann boundary condition, and the electric field at the exit caused by electron lost.

1. Poisson equation solver functions added,
   solvePoissonEquation_2Neumann_jl implements Neumann type boundary condition on both side.
   solvePoissonEquation_DN_jl implements Dirichlet type boundary condition on LHS, and Neumann boundry condition on RHS.
   ![微信图片_20240324113952](https://github.com/JerryGHT04/AmbipolarThrusterPIC/assets/162717938/8a50a214-2c64-4b30-b76a-db7ba986a206)
   Inside the function, A and B are modified.

2. Particle remover function modified,
   boundaryConditionsRectangularNoWallLoss_returndE_jl added.
   This new function computes the electric field strength at RHS plane by knowing the 'qArray' of the particle that is removed at the boundary.
   The change in electric field is then:
   dE = (-Ni_lost + Ne_lost)/(electric_constant*Aexit)
   where Ne_lost += qArray (of the electron superpaticle lost) and Ni_lost += qArray (of the ion superpaticle lost)

 3. In main loop, the global variable E2 is incremented by dE each iteration to give the electric field at the boundary.

 4. The simulation volume is revised. It now defines with position parameters X2, X3, X4:
![ec4005d1ab894ffb74535f9a8eb35b9](https://github.com/JerryGHT04/AmbipolarThrusterPIC/assets/162717938/f242937e-86de-406b-90cc-c8eb191bbe8c)

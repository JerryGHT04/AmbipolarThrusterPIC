2024.3.8
This version should correctly capture the Neumann boundary condition, and the electric field at the exit caused by electron lost.

1. Poisson equation solver functions added,
   solvePoissonEquation_2Neumann_jl implements Neumann type boundary condition on both side.
   solvePoissonEquation_DN_jl implements Dirichlet type boundary condition on LHS, and Neumann boundry condition on RHS.
   ![微信图片_20240324113952](https://github.com/JerryGHT04/AmbipolarThrusterPIC/assets/162717938/8a50a214-2c64-4b30-b76a-db7ba986a206)
   

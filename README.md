2024.3.8
This version should correctly capture the Neumann boundary condition, and the electric field at the exit caused by electron lost.

1. Poisson equation solver functions added,
   solvePoissonEquation_2Neumann_jl implements Neumann type boundary condition on both side.
   solvePoissonEquation_DN_jl implements Dirichlet type boundary condition on LHS, and Neumann boundry condition on RHS.
   ![fba19429ed772538ddbed74c9914f48](https://github.com/JerryGHT04/AmbipolarThrusterPIC/assets/162717938/a46ccb13-913c-47b1-8bbe-1edb12466de0)

2. Particle remover function modified,
   boundaryConditionsRectangularNoWallLoss_returndE_jl added.
   This new function computes the electric field strength at RHS plane by knowing the 'qArray' of the particle that is removed at the boundary.
   The change in electric field is then:
   dE = (-Qi_lost + Qe_lost)/(electric_constant*Aexit)

 3. In main loop, the global variable E2 is incremented by dE each iteration to give the electric field at the boundary.

 4. The simulation volume is revised. It now defines with position parameters X2, X3, X4:
![ec4005d1ab894ffb74535f9a8eb35b9](https://github.com/JerryGHT04/AmbipolarThrusterPIC/assets/162717938/f242937e-86de-406b-90cc-c8eb191bbe8c)

 5. Thrust and ISP are found using following formula:
    ![f60264a461c52e1f5f5eb79aaa79014](https://github.com/JerryGHT04/AmbipolarThrusterPIC/assets/162717938/c091fe9d-8d5c-4648-a078-bf69e940dfc2)

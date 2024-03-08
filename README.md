# AmbipolarThrusterPIC
Code originally created by Dr Aaron Knoll.

This Particle-in-cell simulation employs 1D Poisson's equation to simulate a thruster worked by ambipolar diffusion. The electrons produced by electron source are accelerated by a cyclotron to high speed and collide with neutral gas to produce plasma. The electrons lost at a faster rate results in a net electric field to accelerate ions.

The Poisson equation to determine potential is: 
$\[
\nabla^2 \Phi = -\frac{\rho}{\epsilon_0}
\]$

and it is discretised in X-direction as:
${\left(\phi_{i-1}-2\phi_i+\phi_{i+1}\right)}/{{dx}^2}={-e\left(n_{ion,i}-n_{electron,i}\right)}/{\varepsilon}$

where n is the number density.

The method employs the concept of super-particle which represents a cluster of real particles. The super-size is defined as: real particle number / super particle number
At each time step the Poisson's equation is solved and the particle motion is updated by the equation of force: $m\vec{v}=q\left(\vec{E}+\vec{v}\times\vec{B}\right)$

The boundary condition for the Poisson's equation will be a Neumann type.

By taking the whole device as a whole, an electric field will be induced by ions and electrons being lost at the exit. The ambipolar electric field hence acts as the Neumann boundary condition.
![911afcac75b86f64cdf8a75bf34c010](https://github.com/JerryGHT04/AmbipolarThrusterPIC/assets/162717938/86a609e5-9d5b-433a-8fba-27111497be80)

In this version, ion source rate = electron source rate. No collision is considered.

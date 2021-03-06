# PyNum4Dummies

## Aims
This project consists in a library `solver_tools` and a bunch of python scripts.

The library aims at 
* solving "easy" 1d partial differential equations (pdes), 
* controlling the pdes to reach a user-defined objective (e.g. suppression of oscillations)
* doing data-assimilation (i.e., calibrate model parameters or initial conditions)
for education and research.


### Examples of pdes:
* The heat equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/458569e3907f54a425e4fecadb138e1f.svg?invert_in_darkmode" align=middle width=73.767705pt height=33.42009pt/>
* Burger equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ec7768a23e0ca2a447be1cc33a099882.svg?invert_in_darkmode" align=middle width=78.92775pt height=28.86675pt/>
* The advection equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/19a5775f98b29fc2720f13ec6f7851aa.svg?invert_in_darkmode" align=middle width=76.63986pt height=28.86675pt/>
### Data assimilation ?
Forecast models usually contain unknown parameters. 
It can be 
* the weather
* the rate of propagation of a disease
* some economic activities
Practically speaking, these parameters may be 
* the model's initial conditions ("<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/2f996063a8ab238f2f1705ee4159938f.svg?invert_in_darkmode" align=middle width=74.73378pt height=24.56553pt/>")
* the model's boundary conditions ("<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/fe83da44c2fee65b461a7a79acf354e6.svg?invert_in_darkmode" align=middle width=74.73378pt height=24.56553pt/>" and "<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/2dae7eebba96e1bf14210987c3498007.svg?invert_in_darkmode" align=middle width=77.690745pt height=24.56553pt/>")
* the model's parameters (<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.537065pt height=14.10255pt/> in the heat equation)

## Files
The project consists in a library and a collection of scripts that illustrates it.


### The libraries 
`solver_tools.py`

It contains:
* a function to integrate the equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/940c85b13e2fb5a4a59056b59c583cf0.svg?invert_in_darkmode" align=middle width=129.80715pt height=28.86675pt/>
* discretization of:
  * space derivative: <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/7da93b46f7712a0110dbfb9d2035e68c.svg?invert_in_darkmode" align=middle width=125.696175pt height=28.86675pt/>
    - LUD
    - upwind
  * second space derivative <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/cb5f9f5c88b90ad2366ca70548b60f8d.svg?invert_in_darkmode" align=middle width=107.509545pt height=33.42009pt/>
    - central_scheme
* cfl (compute and print the cfl number)
* functions to compute the adjoint of the pdes in order to compute gradient of possibly complex costs functions
  - data assimilation (identify initial conditions that allows to fit the model on the data available)
  - model assimilation (identify model parameters that allows to fit the model on the data available)

Other functions will be implemented, to derive reduced order models (ROM) of the equation, as well as computing the adjoint variables (see later).

### The scripts
The various scripts illustrate the use of the library:
* `script_optimal_control.py` - optimal control: it identifies the forcing so the solution matches a criterion
* `script_assimilation.py` - data assimilation (model and/or CI)
* `script_advection.py` - advection equation
* `script_test_accuracy.py` - test the accuracy of the solver
* `script_unsteady_bc.py` - illlustrates unsteady boundary conditions
* `script_burgers_inviscid.py` - inviscid burger equation
* `script_viscous_burgers.py` - viscous burger equation
* `script_operators.py` - general equation

# Things it can solve
- [x] pdes 1D
- [ ] pdes 2D/3D
- [x] optimal control
- [x] assimilation of parameters
- [x] assimilation of initial conditions
- [x] both
- [x] dirichlet conditions (even time dependant)
- [x] periodic bdc
- [ ] neumann (implemented but not tested)

# Todo:
- [x] add adjoint
- [x] add assimilation
- [x] add optimal control
- [x] Bug in OC: <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/1e8d876be2cd1ebb3f040126884ada03.svg?invert_in_darkmode" align=middle width=60.131775pt height=22.74591pt/> are unstable on BC with dirichlet
- [x] Unstable diffusivity in adjoint (solved by smoothering...)
- [ ] stencils for the pre-written adjoint adjoint ? That would accelerate a lot the code
- [ ] use of solver_tools instead of solver_matrix_tools when needed ?
- [x] advection: instabilities in LUD when too stiff ? Flux limiter ?
- [x] add <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/9a71e411a99b8df0c66c265d4cdf85da.svg?invert_in_darkmode" align=middle width=30.97413pt height=22.74591pt/>/ <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/23ae0d83f5393e9d38dc1ab12c4973d9.svg?invert_in_darkmode" align=middle width=38.40078pt height=22.74591pt/> for KdW or Kuramoto–Sivashinsky equations ?
- [ ] but instable like diffusivity
- [ ] better presentation here
- [ ] Remove useless scripts and improve compatibility between solver_tools and solver_matrix_tools

# Explanations
## Provided discretization schemes
### Advection
LUD is the Linear Upwind Differencing scheme. It has significantly less dissipation than the regular upwind scheme.
It is used to discretize, in space, the following advection-type operator:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/350014df326d63640dafa3e073a410be.svg?invert_in_darkmode" align=middle width=29.680035pt height=33.769395pt/></p> 

<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode" align=middle width=8.656725pt height=14.10255pt/> does not have to be a constant: it can actually be <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6d32ab707a69e1b60bcb0d5cdbb2ddac.svg?invert_in_darkmode" align=middle width=43.972335pt height=24.56553pt/>.

### Diffusion
central difference scheme is used to discretize, in space, the following diffusion-type operator:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/2a2582121c1c1f8ca35d7b479e2f4413.svg?invert_in_darkmode" align=middle width=47.60283pt height=35.749725pt/></p> 

### Solving a Partial Differential Equation
Let's use a simple example here, the Burger equation:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/1cb2e84f862508d152ba6482e61ffd2f.svg?invert_in_darkmode" align=middle width=240.15585pt height=35.749725pt/></p>

> <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/84cc939597f3eec200843a2fc8830732.svg?invert_in_darkmode" align=middle width=13.39734pt height=22.38192pt/> can be constructed as a list of operators: <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/219ecac31248ffcfb5d6a0fbf233654c.svg?invert_in_darkmode" align=middle width=86.830755pt height=24.65793pt/>.

In the present approach, we chose to separate the time derivative operator from the spatial operators:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/f0bd8b387fe19e78c66b6a63a3164852.svg?invert_in_darkmode" align=middle width=245.256pt height=35.749725pt/></p>

with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/3aec98a126651a10d31427027226826d.svg?invert_in_darkmode" align=middle width=89.35014pt height=28.86675pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/84481e6a2e7547617ef975ff490f37ba.svg?invert_in_darkmode" align=middle width=82.78941pt height=33.42009pt/>. Clearly, these operators can be nonlinear.

> The list of RHS operators <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/1d1c652dbcc962aaf99125ab3d937a38.svg?invert_in_darkmode" align=middle width=53.427pt height=24.56553pt/> will be passed to `integration_forward`, in order to solve the pde.


### Computing the gradient of the cost functional

We want to look at the problem depending on some parameters:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6b5d87a347c821614f8b55c722508a6a.svg?invert_in_darkmode" align=middle width=169.6233pt height=40.675965pt/></p>
s.t. the physics:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d200567cf84acf5ab2731c66fe8752a1.svg?invert_in_darkmode" align=middle width=110.66583pt height=16.376943pt/></p>

> The physics is solved using the function `integration_forward`

For instance, considering <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/35e5cdaf66f3d610c421c0c18a4d656f.svg?invert_in_darkmode" align=middle width=83.34876pt height=24.56553pt/>, and the inviscid Burgers' equation, we have

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/049cefa42bd6a8e61e29850050f0b6f5.svg?invert_in_darkmode" align=middle width=226.5021pt height=33.769395pt/></p>


If the cost function means fitting the model on available data <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d4378ba898213096600125929214f90a.svg?invert_in_darkmode" align=middle width=27.270705pt height=24.56553pt/>, then, one have:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/eeb1823bd9e58f2bb948adf7ae7eec3f.svg?invert_in_darkmode" align=middle width=173.59815pt height=16.376943pt/></p>


We are also considering that the initial conditions are (potentially) related to the parameters q with:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6b1545e49da27eecc71689bc4caec24e.svg?invert_in_darkmode" align=middle width=96.687855pt height=16.376943pt/></p>

> <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.6816575pt height=21.60213pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995pt height=14.10255pt/> do not have to be constructed. In practice, only their partial derivatives w.r.t. <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg?invert_in_darkmode" align=middle width=7.8985335pt height=14.10255pt/> will be needed.

For instance, if the initial conditions are actually the parameter <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/db690fdb4ebf32ef3722ac5b7d64d136.svg?invert_in_darkmode" align=middle width=15.054105pt height=14.10255pt/>:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/2b76a93e1a4ea33f591cbc61e9da7cfd.svg?invert_in_darkmode" align=middle width=193.90635pt height=16.376943pt/></p>

We want to find the argmin of J.
This formalism is usefull in numerous situations, e.g.:
* to identify the parameters of a model that will match some data
* some initial conditions that will reproduce as good as possible the provided data.
* optimal control, so that <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> will reach a given state or follow a given trajectory.
* more generally any user-defined constrains expressed in the form of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.65636pt height=22.38192pt/>.

Identifying the minimum of the <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.65636pt height=22.38192pt/> relies on the gradient of the functional with respect to the parameters:
<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8b3f544e88b0a34f6b2c1dff52a7660e.svg?invert_in_darkmode" align=middle width=77.02893pt height=28.61199pt/>. 

Estimation of the gradient with finite differentiation is out of reach if the size of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg?invert_in_darkmode" align=middle width=7.8985335pt height=14.10255pt/> is large (if <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/eee6665ce62bddc33409a908ab9ef854.svg?invert_in_darkmode" align=middle width=16.24392pt height=14.10255pt/> is the size of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d5c18a8ca1894fd3a7d25f242cbe8890.svg?invert_in_darkmode" align=middle width=7.8985335pt height=14.10255pt/>, then <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8673258717c25292341c86b159af459b.svg?invert_in_darkmode" align=middle width=24.432375pt height=21.10812pt/> evaluation of the system are needed).

For that, one can introduce the Lagrangian <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/a7e36e2400be2dc4b9d95a945ac08ad3.svg?invert_in_darkmode" align=middle width=148.428555pt height=47.66652pt/>, associated with the two Lagrange parameters <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> (variables are dropped for visibility):

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/368ec6b8ab07e536747b028f367e2e3b.svg?invert_in_darkmode" align=middle width=206.37375pt height=41.170305pt/></p>

where <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0103700fa3db36d24bf7a94ad80ae6a5.svg?invert_in_darkmode" align=middle width=9.498225pt height=27.59823pt/> is the transpose operator. 
Naturally, both <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/02578075daf3b73a16fbd736847ed846.svg?invert_in_darkmode" align=middle width=9.375135pt height=21.87504pt/> are considered as variables.


The gradient <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/bd6c3bd24e54898b15c4e707f007eea6.svg?invert_in_darkmode" align=middle width=76.077045pt height=28.61199pt/> of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/f240a2045bc52f353f0dfa6a0054016c.svg?invert_in_darkmode" align=middle width=11.29557pt height=22.38192pt/> is
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d8555c3f42e49376b97626e02ba69eb8.svg?invert_in_darkmode" align=middle width=666.42345pt height=41.170305pt/></p>

Upon optimality, one has <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/52af41d8ee5aa88040ceb890cd74d959.svg?invert_in_darkmode" align=middle width=85.45878pt height=22.38192pt/>.

The term in <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/c2f43910af366cae2d41e81ddd7e93a5.svg?invert_in_darkmode" align=middle width=16.437795pt height=22.74591pt/> cannot be easily estimated. An integration by parts gives:


<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0730187cc4f27608f78cf0efa12b847b.svg?invert_in_darkmode" align=middle width=492.80055pt height=41.121795pt/></p>


The term associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d334a39817444a817c8f52832258f502.svg?invert_in_darkmode" align=middle width=16.437795pt height=22.74591pt/> can now be replaced.

Ordering terms leads to:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6723dbf2d7962d18a7080ca751167a33.svg?invert_in_darkmode" align=middle width=948.4629pt height=42.13704pt/></p>

As <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/84cc939597f3eec200843a2fc8830732.svg?invert_in_darkmode" align=middle width=13.39734pt height=22.38192pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995pt height=14.10255pt/> are null by construction, <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> can be designed specifically to alleviate the computations.

Indeed, proper choices for <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0ae3f8e52e8833a55be05df21031a4fc.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> allow to simplify the expression of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d0401d7fec19d46153d33617ba7d0d7b.svg?invert_in_darkmode" align=middle width=32.114115pt height=22.38192pt/>.

The choice of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/544873064b19aa6af6bc1a0f71eddf75.svg?invert_in_darkmode" align=middle width=64.20282pt height=24.56553pt/> nullifies the term <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ee0fc38ca1173499445649fb428e3dda.svg?invert_in_darkmode" align=middle width=143.40975pt height=47.6718pt/>.
<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0ae3f8e52e8833a55be05df21031a4fc.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> can then be chosen as the solution of the so-called adjoint equation(see Ledimet 1986,Talagrand 1997):

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/b75b4f1675717186200fc810aaea5244.svg?invert_in_darkmode" align=middle width=334.8906pt height=39.30498pt/></p>
integrated backwards in time. 

> This equation is solved using the function `integration_backward`

To do so, linearize <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/84cc939597f3eec200843a2fc8830732.svg?invert_in_darkmode" align=middle width=13.39734pt height=22.38192pt/> around <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> and replace <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> 


Finally, the Lagrange parameter <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/4d01d1168740312c9cbf4c58d10ac5f7.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> is set so that it nullifies the component associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/899d25373b79ec2a7e855b098bf9b9b8.svg?invert_in_darkmode" align=middle width=46.255605pt height=24.56553pt/>:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/297f961a145edf067bea059f26a5e49f.svg?invert_in_darkmode" align=middle width=188.9349pt height=40.274355pt/></p>

Then, computing <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d0401d7fec19d46153d33617ba7d0d7b.svg?invert_in_darkmode" align=middle width=32.114115pt height=22.38192pt/>, hence <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8116b309283a1964c0488ce84f61c376.svg?invert_in_darkmode" align=middle width=31.474905pt height=22.38192pt/>, is achieved by the integration of:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/f07e5a5062a3abc62fcb796a174cad33.svg?invert_in_darkmode" align=middle width=323.18715pt height=41.170305pt/></p>

> This equation is solved using the function `gradient_q`


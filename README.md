# PyNum4Dummies

## Aims
This project consists in a library solver_tools and a bunch of python scripts.

The library aims at solving "easy" 1d partial differential equations (pdes), for education and research.


### Exemples of pdes:
* The heat equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/458569e3907f54a425e4fecadb138e1f.svg?invert_in_darkmode" align=middle width=73.767705pt height=33.42009pt/>
* Burger equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ec7768a23e0ca2a447be1cc33a099882.svg?invert_in_darkmode" align=middle width=78.92775pt height=28.86675pt/>
* The advection equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/19a5775f98b29fc2720f13ec6f7851aa.svg?invert_in_darkmode" align=middle width=76.63986pt height=28.86675pt/>


## Files
The project consists in a library and a collection of scripts that illustrates it.


### The librairie 
solver_tools.py

It contains:
* a function to integrate the equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/940c85b13e2fb5a4a59056b59c583cf0.svg?invert_in_darkmode" align=middle width=129.80715pt height=28.86675pt/>
* discretization of:
  * space derivative: <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/7da93b46f7712a0110dbfb9d2035e68c.svg?invert_in_darkmode" align=middle width=125.696175pt height=28.86675pt/>
    - LUD
    - upwind
  * second space derivative <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/cb5f9f5c88b90ad2366ca70548b60f8d.svg?invert_in_darkmode" align=middle width=107.509545pt height=33.42009pt/>
    - central_scheme
* cfl (compute and print the cfl number)

Other functions will be implemented, to derive reduced order models (ROM) of the equation, as well as computing the adjoint variables (see later).

### The scripts
The various scripts illustrate the use of the library:
* script_advection.py - advection equation
* script_test_accuracy.py - test the accuracy of the solver
* script_unsteady_bc.py - illlustrates unsteady boundary conditions
* script_burgers_inviscid.py - inviscid burger equation
* script_viscous_burgers.py - viscous burger equation
* script_operators.py - general equation


## Todo:
* advection: instabilities in LUD when too stiff ? Flux limiter ?
* adjoint.



# Computing the gradient of the cost functional

We want to look at the problem depending on some parameters:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0d357cfaf830c4e7c0dbc17ab01006b9.svg?invert_in_darkmode" align=middle width=155.940675pt height=41.121795pt/></p>
s.t. the physics:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/16283e6f9f4969cb2a841640a223eb9a.svg?invert_in_darkmode" align=middle width=110.076945pt height=16.376943pt/></p>
and considering that the initial conditions are (potentially) related to the parameters q with:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6b1545e49da27eecc71689bc4caec24e.svg?invert_in_darkmode" align=middle width=96.687855pt height=16.376943pt/></p>

We want to find the argmin of J.

Identifying the minimum of the <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.65636pt height=22.38192pt/> relies on the gradient of the functional with respect to the parameters:
<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8b3f544e88b0a34f6b2c1dff52a7660e.svg?invert_in_darkmode" align=middle width=77.02893pt height=28.61199pt/>. 

Finite differentiation is out of reach if the size of q is large.

For that, one can introduce the Lagrangian <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ca0f3a73abc788c4c397d1c983cc5b31.svg?invert_in_darkmode" align=middle width=11.14542pt height=22.38192pt/>, associated with the two Lagrange parameters <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/>:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/a518e38eaed61e3ae0f0b3b2339d5646.svg?invert_in_darkmode" align=middle width=308.07975pt height=41.616135pt/></p>
where <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0103700fa3db36d24bf7a94ad80ae6a5.svg?invert_in_darkmode" align=middle width=9.498225pt height=27.59823pt/> is the transpose operator. 
Naturally, both <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/02578075daf3b73a16fbd736847ed846.svg?invert_in_darkmode" align=middle width=9.375135pt height=21.87504pt/> are considered as variables.

As <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.80598pt height=22.38192pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995pt height=14.10255pt/> are null by construction, <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> can be designed specifically to alleviate the computations.

The gradient <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/cdcddae0e184a3d241940d91f2a584d1.svg?invert_in_darkmode" align=middle width=75.926895pt height=28.61199pt/> of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ca0f3a73abc788c4c397d1c983cc5b31.svg?invert_in_darkmode" align=middle width=11.14542pt height=22.38192pt/> is
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/08fa9cd6495b8cbafdf4039d351151f5.svg?invert_in_darkmode" align=middle width=667.4877pt height=41.616135pt/></p>

Upon optimality, one has <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8d8571e42d6aa0132bc47923de52b7aa.svg?invert_in_darkmode" align=middle width=85.308795pt height=22.38192pt/>.

The term in <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/c2f43910af366cae2d41e81ddd7e93a5.svg?invert_in_darkmode" align=middle width=16.437795pt height=22.74591pt/> cannot be easily estimated. An integration by parts gives:


<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/76ed7bc4217709bcf3ca756f860aa836.svg?invert_in_darkmode" align=middle width=493.647pt height=41.121795pt/></p>

The term associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d334a39817444a817c8f52832258f502.svg?invert_in_darkmode" align=middle width=16.437795pt height=22.74591pt/> can now be replaced.

Ordering terms leads to:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/833c8c9603b1acb0cee91506a1b05514.svg?invert_in_darkmode" align=middle width=947.7501pt height=42.58287pt/></p>

Proper choices for <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0ae3f8e52e8833a55be05df21031a4fc.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> allow to simplify the expression of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/77f72004c2606e02e7d9060c80ccca7e.svg?invert_in_darkmode" align=middle width=31.963965pt height=22.38192pt/>.
The choice of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/544873064b19aa6af6bc1a0f71eddf75.svg?invert_in_darkmode" align=middle width=64.20282pt height=24.56553pt/> nullifies the term <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/99a630258fff19b09dad026d69902c4e.svg?invert_in_darkmode" align=middle width=143.25399pt height=47.6718pt/>.
<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0ae3f8e52e8833a55be05df21031a4fc.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> can then be chosen as the solution of the so-called adjoint equation,~\cite{Ledimet1986,Talagrand1997}:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/16cd4b56c6e04130965b7a0b2f11f729.svg?invert_in_darkmode" align=middle width=333.12345pt height=39.30498pt/></p>
integrated backwards in time. 

Finally, the Lagrange parameter <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/4d01d1168740312c9cbf4c58d10ac5f7.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> is set so that it nullifies the component associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/899d25373b79ec2a7e855b098bf9b9b8.svg?invert_in_darkmode" align=middle width=46.255605pt height=24.56553pt/>:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ca0d3b418b6e39f36a643cb76db332b8.svg?invert_in_darkmode" align=middle width=188.34255pt height=40.274355pt/></p>

Then, computing <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/77f72004c2606e02e7d9060c80ccca7e.svg?invert_in_darkmode" align=middle width=31.963965pt height=22.38192pt/>, hence <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8116b309283a1964c0488ce84f61c376.svg?invert_in_darkmode" align=middle width=31.474905pt height=22.38192pt/>, is achieved by the integration of:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ada9d358292a2ddd134f8456fade4925.svg?invert_in_darkmode" align=middle width=325.57965pt height=41.616135pt/></p>



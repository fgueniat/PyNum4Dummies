# PyNum4Dummies

This project consists in a library solver_tools and a bunch of python scripts.

The library aims at solving "easy" 1d partial differential equations (pdes), for education and reasearach.
Exemples of these pdes are:
* The heat equation
* Burger equation
* The advection equation

The librairie contains:
* a function to integrate the equation
* discretization of:
	** space derivative:
    	- LUD
     	- upwind
	** second space derivative
    	- central_scheme
* cfl (compute and print the cfl number)

Other functions will be implemented, to derive reduced order models (ROM) of the equation, as well as computing the adjoint variables (see later).

The various scripts illustrate the use of the library.

Todo:
* advection: instabilities in LUD when too stiff ? Flux limiter ?
* adjoint.






Computing the gradient of the cost functional

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

For that, one can introduce the Lagrangian <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ca0f3a73abc788c4c397d1c983cc5b31.svg?invert_in_darkmode" align=middle width=11.14542pt height=22.38192pt/>, associated with the two Lagrange parameters <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/67e1ca79bcc5c927928c07a5ab112c3a.svg?invert_in_darkmode" align=middle width=10.986195pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/c675de927d56f6ad8de50d4b32b3ade7.svg?invert_in_darkmode" align=middle width=11.592735pt height=14.55729pt/>:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/82b542efcb599a5440486816e9e4258c.svg?invert_in_darkmode" align=middle width=325.70835pt height=39.30498pt/></p>
where <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/0103700fa3db36d24bf7a94ad80ae6a5.svg?invert_in_darkmode" align=middle width=9.498225pt height=27.59823pt/> is the transpose operator. 
Naturally, both <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/02578075daf3b73a16fbd736847ed846.svg?invert_in_darkmode" align=middle width=9.375135pt height=21.87504pt/> are considered as variables.

As <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.80598pt height=22.38192pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995pt height=14.10255pt/> are null by construction, <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/67e1ca79bcc5c927928c07a5ab112c3a.svg?invert_in_darkmode" align=middle width=10.986195pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/c675de927d56f6ad8de50d4b32b3ade7.svg?invert_in_darkmode" align=middle width=11.592735pt height=14.55729pt/> can be designed specifically to alleviate the computations.

The gradient <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/cdcddae0e184a3d241940d91f2a584d1.svg?invert_in_darkmode" align=middle width=75.926895pt height=28.61199pt/> of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ca0f3a73abc788c4c397d1c983cc5b31.svg?invert_in_darkmode" align=middle width=11.14542pt height=22.38192pt/> is
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ae63f233c710e6cae88892bf3b66799f.svg?invert_in_darkmode" align=middle width=496.03785pt height=60.7563pt/></p>
Upon optimality, one has <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/d2462d2c4dea68142b3c37c09896a619.svg?invert_in_darkmode" align=middle width=86.336745pt height=22.38192pt/>.

The term in <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/4492a03d19c8a064593f8767411bb166.svg?invert_in_darkmode" align=middle width=16.267185pt height=22.74591pt/> cannot be easily estimated. An integration by parts gives:


<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/22c72f459dc30283c98cb702292c7420.svg?invert_in_darkmode" align=middle width=505.54515pt height=41.121795pt/></p>
The term associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/187b16ff190bebafb8d6f6e7d28ef9e4.svg?invert_in_darkmode" align=middle width=16.87554pt height=22.74591pt/> can now be replaced.

Ordering terms leads to:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/a3d513a8cb1164692c4e51787b62178e.svg?invert_in_darkmode" align=middle width=1053.8451pt height=49.13139pt/></p>

Proper choices for <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/67e1ca79bcc5c927928c07a5ab112c3a.svg?invert_in_darkmode" align=middle width=10.986195pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/c675de927d56f6ad8de50d4b32b3ade7.svg?invert_in_darkmode" align=middle width=11.592735pt height=14.55729pt/> allow to simplify the expression of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/77f72004c2606e02e7d9060c80ccca7e.svg?invert_in_darkmode" align=middle width=31.963965pt height=22.38192pt/>.
The choice of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/b566a58420e686c547370f31699460d9.svg?invert_in_darkmode" align=middle width=54.167355pt height=22.74591pt/> nullifies the term <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/7d081f8eefa17c2b24e11e604da3992f.svg?invert_in_darkmode" align=middle width=145.720245pt height=47.6718pt/>.
<img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/67e1ca79bcc5c927928c07a5ab112c3a.svg?invert_in_darkmode" align=middle width=10.986195pt height=22.74591pt/> can then be chosen as the solution of the so-called adjoint equation,~\cite{Ledimet1986,Talagrand1997}:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/ca2366de34a7542f11469fe29bdf0335.svg?invert_in_darkmode" align=middle width=345.85155pt height=80.401365pt/></p>
integrated backwards in time. 
Finally, the Lagrange parameter <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/c675de927d56f6ad8de50d4b32b3ade7.svg?invert_in_darkmode" align=middle width=11.592735pt height=14.55729pt/> is set so that it nullifies the component associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/eab061cb04530c077526ab0ef8b7bfd8.svg?invert_in_darkmode" align=middle width=40.34184pt height=22.74591pt/>:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/200978df76de0ae9c6fad4401e659c68.svg?invert_in_darkmode" align=middle width=168.11685pt height=30.411315pt/></p>
Then, computing <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/77f72004c2606e02e7d9060c80ccca7e.svg?invert_in_darkmode" align=middle width=31.963965pt height=22.38192pt/>, hence <img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/8116b309283a1964c0488ce84f61c376.svg?invert_in_darkmode" align=middle width=31.474905pt height=22.38192pt/>, is achieved by the integration of:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/master/svgs/a3f2d072ff7c3801eae2876b8f29cf4b.svg?invert_in_darkmode" align=middle width=301.8213pt height=39.30498pt/></p>



# PyNum4Dummies

This project consists in a library solver_tools and a bunch of python scripts.

The library aims at solving "easy" 1d partial differential equations (pdes), for education and reasearach.
Exemples of these pdes are:
* The heat equation <p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/99bf311583d305ed37ca20321fd06a39.svg?invert_in_darkmode" align=middle width=81.82812pt height=35.749725pt/></p>
* Burger equation <p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/034ecdc5c99727e321e0c207b2f2b522.svg?invert_in_darkmode" align=middle width=126.73815pt height=33.769395pt/></p>
* The advection equation <p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/8c06bc4a89485084ada1a2d1b87a7c9f.svg?invert_in_darkmode" align=middle width=124.45026pt height=33.769395pt/></p>

The librairie contains:
* a function to integrate the equation <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/940c85b13e2fb5a4a59056b59c583cf0.svg?invert_in_darkmode" align=middle width=129.80715pt height=28.86675pt/>
* discretization of:
	** space derivative: <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/7da93b46f7712a0110dbfb9d2035e68c.svg?invert_in_darkmode" align=middle width=125.696175pt height=28.86675pt/>
    	- LUD
     	- upwind
	** second space derivative <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/cb5f9f5c88b90ad2366ca70548b60f8d.svg?invert_in_darkmode" align=middle width=107.509545pt height=33.42009pt/>
    	- central_scheme
* cfl (compute and print the cfl number)

Other functions will be implemented, to derive reduced order models (ROM) of the equation, as well as computing the adjoint variables (see later).

The various scripts illustrate the use of the library.

Todo:
* advection: instabilities in LUD when too stiff ? Flux limiter ?
* adjoint.






Computing the gradient of the cost functional

We want to look at the problem depending on some parameters:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/0d357cfaf830c4e7c0dbc17ab01006b9.svg?invert_in_darkmode" align=middle width=155.940675pt height=41.121795pt/></p>
s.t. the physics:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/16283e6f9f4969cb2a841640a223eb9a.svg?invert_in_darkmode" align=middle width=110.076945pt height=16.376943pt/></p>
and considering that the initial conditions are (potentially) related to the parameters q with:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/6b1545e49da27eecc71689bc4caec24e.svg?invert_in_darkmode" align=middle width=96.687855pt height=16.376943pt/></p>

We want to find the argmin of J.

Identifying the minimum of the <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.65636pt height=22.38192pt/> relies on the gradient of the functional with respect to the parameters:
<img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/8b3f544e88b0a34f6b2c1dff52a7660e.svg?invert_in_darkmode" align=middle width=77.02893pt height=28.61199pt/>. 

Finite differentiation is out of reach if the size of q is large.

For that, one can introduce the Lagrangian <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/ca0f3a73abc788c4c397d1c983cc5b31.svg?invert_in_darkmode" align=middle width=11.14542pt height=22.38192pt/>, associated with the two Lagrange parameters <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/a7db5ae06035b59a72eaab8676b76a37.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/727ea98f01155866832ba9919f928160.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/>:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/a518e38eaed61e3ae0f0b3b2339d5646.svg?invert_in_darkmode" align=middle width=308.07975pt height=41.616135pt/></p>
where <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/0103700fa3db36d24bf7a94ad80ae6a5.svg?invert_in_darkmode" align=middle width=9.498225pt height=27.59823pt/> is the transpose operator. 
Naturally, both <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.375135pt height=14.10255pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/02578075daf3b73a16fbd736847ed846.svg?invert_in_darkmode" align=middle width=9.375135pt height=21.87504pt/> are considered as variables.

As <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/b8bc815b5e9d5177af01fd4d3d3c2f10.svg?invert_in_darkmode" align=middle width=12.80598pt height=22.38192pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.398995pt height=14.10255pt/> are null by construction, <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/a7db5ae06035b59a72eaab8676b76a37.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/727ea98f01155866832ba9919f928160.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> can be designed specifically to alleviate the computations.

The gradient <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/cdcddae0e184a3d241940d91f2a584d1.svg?invert_in_darkmode" align=middle width=75.926895pt height=28.61199pt/> of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/ca0f3a73abc788c4c397d1c983cc5b31.svg?invert_in_darkmode" align=middle width=11.14542pt height=22.38192pt/> is
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/922a69023e3e7b04fcd94fed6ad547c5.svg?invert_in_darkmode" align=middle width=667.4877pt height=41.616135pt/></p>
Upon optimality, one has <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/1b3d5974d13f25fd5135a383f4fcba69.svg?invert_in_darkmode" align=middle width=85.308795pt height=22.38192pt/>.

The term in <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/4492a03d19c8a064593f8767411bb166.svg?invert_in_darkmode" align=middle width=16.267185pt height=22.74591pt/> cannot be easily estimated. An integration by parts gives:


<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/79e3554dc3d7a585a5455122efd64082.svg?invert_in_darkmode" align=middle width=493.47705pt height=41.121795pt/></p>
The term associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/187b16ff190bebafb8d6f6e7d28ef9e4.svg?invert_in_darkmode" align=middle width=16.87554pt height=22.74591pt/> can now be replaced.

Ordering terms leads to:

<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/f09532a8f24f9c9f0ae919728daf82ad.svg?invert_in_darkmode" align=middle width=1035.8667pt height=49.13139pt/></p>

Proper choices for <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/a7db5ae06035b59a72eaab8676b76a37.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> and <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/727ea98f01155866832ba9919f928160.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> allow to simplify the expression of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/77f72004c2606e02e7d9060c80ccca7e.svg?invert_in_darkmode" align=middle width=31.963965pt height=22.38192pt/>.
The choice of <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/594e597ddc551f31cadbe30408001056.svg?invert_in_darkmode" align=middle width=51.46515pt height=22.74591pt/> nullifies the term <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/74011edf4370094264c3865995beec7e.svg?invert_in_darkmode" align=middle width=143.25399pt height=47.6718pt/>.
<img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/a7db5ae06035b59a72eaab8676b76a37.svg?invert_in_darkmode" align=middle width=9.553335pt height=22.74591pt/> can then be chosen as the solution of the so-called adjoint equation,~\cite{Ledimet1986,Talagrand1997}:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/850c76e26f0744f54aa8afca9013aabe.svg?invert_in_darkmode" align=middle width=340.20195pt height=80.401365pt/></p>
integrated backwards in time. 
Finally, the Lagrange parameter <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/727ea98f01155866832ba9919f928160.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> is set so that it nullifies the component associated with <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/eab061cb04530c077526ab0ef8b7bfd8.svg?invert_in_darkmode" align=middle width=40.34184pt height=22.74591pt/>:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/9aca4681cd3c972fe11ceeed38d17000.svg?invert_in_darkmode" align=middle width=163.68462pt height=21.196395pt/></p>
Then, computing <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/77f72004c2606e02e7d9060c80ccca7e.svg?invert_in_darkmode" align=middle width=31.963965pt height=22.38192pt/>, hence <img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/8116b309283a1964c0488ce84f61c376.svg?invert_in_darkmode" align=middle width=31.474905pt height=22.38192pt/>, is achieved by the integration of:
<p align="center"><img src="https://rawgit.com/fgueniat/PyNum4Dummies/None/svgs/df3658206d982d2298e7d2fe8fe47555.svg?invert_in_darkmode" align=middle width=294.9309pt height=39.30498pt/></p>



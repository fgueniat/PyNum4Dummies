# PyNum4Dummies
This project consists in a library solver_tools and python script.
The library aims at solving burgers and easy 1d pdes, for education and reasearach.
It contains 
* an integrator (integration)
* discretization of:
	** space derivative:
    	- LUD
     	- upwind
	** second space derivative
    	- central_scheme
* cfl (compute and print the cfl number)

Other functions to come to compute the adjoint.

The script illustrates the use of the library.

Todo:
* advection: instabilities in LUD when too stiff ?
* adjoint.




import numpy as np
'''
library for solving burgers and easy 1d pde.
contains 
an integrator (integration)
discretization of:
space derivative:
    LUD
    upwind
second space derivative
    central_scheme
cfl (compute and print the cfl number)
'''

param_n_ghost = 10


def LUD(a,u,x):
    '''
    discretization of $ a \frac{\partial u} \{partial x} $ 
    Linear Upwind Differencing
    2nd order upwind: less diffusion than regular upwind
    the minus signs come from it is treated as a RHS rather than a LHS
    a is the term in front of the spatial derivative (it can be a function of x or u)
    u is the quantity to be discretized
    x is the space
    '''
    if np.isscalar(a):
        ap = np.ones(u.shape) * np.maximum(-a,0)
        am = np.ones(u.shape) * np.minimum(-a,0)
    else:
        ap = np.maximum(-a,0)
        am = np.minimum(-a,0)
    dudx = np.zeros(u.shape)
    dudx[2:-2]  = ap[2:-2] * (  3.*u[2:-2] - 4.*u[1:-3] +    u[:-4]  ) / (2.* (x[3:-1] - x[1:-3])/2. ) 
    dudx[2:-2] += am[2:-2] * (    -u[4:]   + 4.*u[3:-1] - 3.*u[2:-2] ) / (2.* (x[3:-1] - x[1:-3])/2. )
    return -dudx


def upwind(a,u,x):
    '''
    discretization of $ a \frac{\partial u} \{partial x} $ 
    upwind scheme
    the minus signs come from it is treated as a RHS rather than a LHS
    a is the term in front of the spatial derivative (it can be a function of x or u)
    u is the quantity to be discretized
    x is the space
    '''
    if np.isscalar(a):
        ap = np.ones(u.shape) * np.maximum(-a,0)
        am = np.ones(u.shape) * np.minimum(-a,0)
    else:
        ap = np.maximum(-a,0)
        am = np.minimum(-a,0)
    dudx = np.zeros(u.shape)
    dudx[1:-1] = ap[1:-1]*(u[1:-1] - u[0:-2])/(x[1:-1] - x[0:-2]) + am[1:-1]*(u[2:] - u[1:-1]) / (x[2:] - x[1:-1])
    return -dudx

def central_scheme(u,x):
    '''
    discretization of $ \frac{\partial^2 u} \{partial x^2} $ 
    central difference scheme scheme
    u is the quantity to be discretized
    x is the space
    '''
    d2udx2 = np.zeros(u.shape)
    d2udx2[1:-1] = ( u[2:] + u[:-2] - 2.* u[1:-1]) / (2.* ( ( x[2:]-x[:-2] )/2. ) **2. )
    return d2udx2


def cfl(u,x,dt):
    '''
    print the cfl number
    '''
    print (np.max(u)*dt/np.max(np.diff(x)))

def ghost(u,x,bc_type='periodic',bcs = None ,n_g=param_n_ghost):
    '''
    extend the space for imposing the boundary conditions
    u is the quantity to be discretized
    x is the space
    bc_type is the type of BC: periodic, neumann (flux) or dirichlet (fixed BC)
    bcs is the values (flux in Neumann's case, fixed values in dirichlet's case)
    n_g is the size of the sponge zone
    '''
    # first the space
    x = np.r_[
                x[0] - (x[1] - x[0]) * np.arange(1,n_g+1)[::-1], 
                x , 
                x[-1] + (x[-1] - x[-2]) * np.arange(1,n_g+1)
            ]
    #
    if bc_type == 'periodic':
        u = np.r_[
                    u[-n_g-1:-1], 
                    u[:-1] , 
                    u[:n_g+1]
                ]
    #
    if bc_type == 'dirichlet': # dirichlet
        u = np.r_[
                    bcs[0]*np.ones(n_g), 
                    u ,
                    bcs[1]*np.ones(n_g)
                    ]
    #    
    if bc_type == 'neumann':
        dxl = x[0] - x[1]
        dxr = x[-1] - x[-2]
        u = np.r_[
                u[0] - bcs[0] * dxr * np.arange(1,n_g+1)[::-1], 
                    u ,  
                    u[-1] + bcs[1] * dxr * np.arange(1,n_g+1) 
                ]
    #
    return u,x

def deghost(u,x,n_g=param_n_ghost):
    '''
    remove the sponge zone
    '''
    u = u[n_g:-n_g]
    x = x[n_g:-n_g]
    return u,x


def RHS(u,x,t,dt,operators = None ):
    '''
    add all the terms coming from the operators.
    u is the quantity to solve
    x is the space
    t is the current time
    dt is the step time
    operators is a list of operators, i.e. functions accepting u,x,t,dt as parameters.
    It represents the problem to be solve.
    e.g., for the advection problem with the velocity c:
    advection = lambda u,x,t,dt: c * LUD(u,x,t,dt)
    operators = [[advection]]
    '''
    rhs = np.zeros(u.shape)
    for operator in operators:
        rhs += operator(u,x,t,dt)
    return rhs


def integration(u, x, t, dt, operators = None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False):
    '''
    bind to integration_forward
    '''
    return integration_forward(u,x,t,dt, operators = operators, method = method, bc_type = bc_type, bcs = bcs,n_g = n_g, return_rhs = return_rhs)

def integration_forward(u, x, t, dt, operators = None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False):
    '''
    Integrate a problem.
    u is the quantity to solve
    x is the space
    t is the current time
    dt is the step time
    operators is a list of operators, i.e. functions accepting u,x,t,dt as parameters.
    It represents the problem to be solved.
    e.g., for the advection problem with the velocity c:
    advection = lambda u,x,t,dt: c * LUD(u,x,t,dt)
    operators = [[advection]]
    method is the integration method. Current choice among 'RK4' and 'EF'.
    bc_type is the type of BC: periodic, neumann (flux) or dirichlet (fixed BC)
    bcs is the values (flux in Neumann's case, fixed values in dirichlet's case)
    n_g is the size of the sponge zone
    return_rhs: True or False. 
    '''
    #
    if operators is None: 
        print('no problem provided')
        return -1
    #
    if method == 'EF': #Euler
        u,x = ghost(u,x, n_g = n_g)
        rhs = dt * RHS(u,x,t,dt,operators = operators)
        u = u + rhs
        u,x = deghost(u,x)
    #
    if method == 'RK4':
        u,x = ghost(u,x,bc_type = bc_type,bcs = bcs)
        f = lambda u,t: RHS(u,x,t,dt,operators = operators)
        rhs = RK4(f, u, t, dt)
        u = u+rhs
        u,x = deghost(u,x,n_g = n_g)
    #
    if return_rhs is True:
        return u,rhs
    else:
        return u


def RK4(f,x,t,dt):
    k1 = dt * f(x,t)
    k2 = dt * f(x+k1/2.,t+dt/2.)
    k3 = dt * f(x+k2/2.,t+dt/2.)
    k4 = dt * f(x+k3,t+dt)
    return  (k1 + 2.*k2 + 2.*k3 + k4)/6.

def integration_backward_adjoint(u,x,t,dt, lambda_adjoint, operators_du, operators_dup, operators_dtdu, cost_function_du, method = 'RK4', n_g = param_n_ghost):
    '''
    Integrate the adjoint problem.
    lambda_adjoint is the quantity to solve
    u is the solution at time t
    x is the space
    t is the current time
    dt is the step time
    operators_ are lists of operators, i.e. functions accepting u,x,t,dt as parameters.
    It represents the problem to be solved.
    du means partial derivative wrt u.
    dup means partial derivative wrt du/dt.
    dtdu means partial derivative wrt time then to u.

    method is the integration method. Current choice among 'RK4' and 'EF'.
    n_g is the size of the sponge zone
    '''

    pass



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

param_n_ghost = 16


def LUD_mat(a,u,x):
    '''
    discretization of $ a \frac{\partial u} \{partial x} $ 
    Linear Upwind Differencing
    2nd order upwind: less diffusion than regular upwind
    the minus signs come from it is treated as a RHS rather than a LHS
    a is the term in front of the spatial derivative (it can be a function of x or u)
    u is the quantity to be discretized
    x is the space
    '''
    A = np.zeros((u.size,u.size))
    if np.isscalar(a):
        ap = np.ones(u.shape) * np.maximum(-a,0)
        am = np.ones(u.shape) * np.minimum(-a,0)
    else:
        ap = np.maximum(-a,0)
        am = np.minimum(-a,0)
    dx = (x[3:-1] - x[1:-3])/2.
    dx = np.r_[dx[0],dx[0],dx,dx[-1],dx[-1]]
    v = np.ones(u.shape)
    # ui
    A +=   np.diag( ap * 3. * v/(2.* dx) ) 
    # ui-1
    A += - np.diag( ap[1:] * 4. * v[:-1]/(2.* dx[1:]), -1 )
    # ui-2
    A +=   np.diag( ap[2:] * v[:-2]/(2.* dx[2:] ), -2 )
    # ui
    A += - np.diag( am * 3. * v/ (2.* dx ) ) 
    # ui+1
    A +=   np.diag( am[:-1] * 4. * v[:-1]/(2.* dx[:-1] ),1 )
    # ui+2
    A += - np.diag( am[:-2] * v[:-2]/(2.* dx[:-2] ), 2 )
    return -A



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

def diffusion_mat(u,x):
    '''
    discretization of $ \frac{\partial^2 u} \{partial x^2} $ 
    central difference scheme scheme
    u is the quantity to be discretized
    x is the space
    '''
    d2udx2 = np.zeros(u.shape)
    A = np.zeros((u.size,u.size))
    dx = (x[2:] - x[:-2])/2.
    dx = np.r_[dx[0],dx,dx[-1]]
    v = np.ones(u.shape) / (2. * dx**2)
    #ui-1
    A += np.diag(v[1:],-1) 
    #ui
    A += -2.*np.diag(v) 
    #ui+1
    A += np.diag(v[1:], 1) 
    return A


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

def RHS_mat(u,x,t,dt,operators = None ):
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
    A = np.zeros((u.size,u.size))
    for operator in operators:
        A += operator(u,x,t,dt)
    return A




def RHS_mat_adjoint(lambda_,u,x,t,dt,operators = None ):
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
    A = np.zeros((lambda_.size,lambda_.size))
    for operator in operators:
        A += operator(lambda_,u,x,t,dt)
    return A




def integration(u, x, t, dt, operators = None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False, Full_Operator = None, return_operator = False):
    '''
    bind to integration_forward_mat
    '''
    return integration_forward_mat(u,x,t,dt, operators = operators, method = method, bc_type = bc_type, bcs = bcs,n_g = n_g, return_rhs = return_rhs, Full_Operator = Full_Operator, return_operator = return_operator)



def integration_forward_mat(u, x, t, dt, operators = None, rhs_forcing = None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False,Full_Operator=None, return_operator = False):
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
    rhs_forcing is a function describing the rhs forcing.
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
    u,x = ghost(u,x,bc_type = bc_type,bcs = bcs)
    #
    if method == 'EF': #Euler
        if Full_Operator is None:
            Full_Operator = RHS_mat(u,x,t,dt,operators = operators)
        #
        A = np.eye(u.size) + dt * Full_Operator
        u = np.dot(Full_Operator,u) + dt * rhs_forcing(u,x,t,dt)
    #
    if rhs_forcing is None: 
        rhs_forcing = lambda u,x,t,dt: np.zeros(x.size)
    #
    if method == 'RK4':
        if Full_Operator is None:
            f = lambda u,t,p: np.dot( RHS_mat(u,p[0],t,p[1],operators = p[2]) ,u) + rhs_forcing(u,p[0],t,p[1])
            rhs = RK4(f, u, t, dt,p=[x,dt,operators])
        else: 
            f = lambda u,t,p: np.dot(Full_Operator,u) + dt * rhs_forcing(u,p[0],t,p[1])
            rhs = RK4(f, u, t, dt,p=[x,dt])
        #
        u = u+rhs
    u,x = deghost(u,x,n_g = n_g)
    #
    if return_rhs is True:
        u = (u,rhs)
    if return_operator is True:
        if return_rhs is True:
            u = u + (Full_Operator,)
        else:
            u = (u,Full_Operator)
        #
    #
    #
    return u


def RK4(f,x,t,dt,p=None):
    if p is None: ff = lambda x,t,p: f(x,t)
    else: ff = lambda x,t,p: f(x,t,p)
    k1 = dt * ff(x,t,p)
    k2 = dt * ff(x+k1/2.,t+dt/2.,p)
    k3 = dt * ff(x+k2/2.,t+dt/2.,p)
    k4 = dt * ff(x+k3,t+dt,p)
    return  (k1 + 2.*k2 + 2.*k3 + k4)/6.


def integration_backward_mat(lambda_,u, x, t, dt, p, operators = None, rhs_forcing = None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False,Full_Operator=None, return_operator = False):
    '''
    Integrate the adjoint problem.
    lambda_ is the quantity to solve
    u is the quantity solved
    x is the space
    t is the current time
    dt is the step time (negative!)
    p is the parameters for rhs_forcing (i.e. dj/du)
    operators is a list of operators, i.e. functions accepting u,x,t,dt as parameters.
    It represents the problem to be solved.
    e.g., for the advection problem with the velocity c:
    advection = lambda u,x,t,dt: c * LUD(u,x,t,dt)
    operators = [[advection]]
    rhs_forcing is a function describing the rhs forcing.
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
    lambda_,_ = ghost(lambda_,x,bc_type = bc_type,bcs = bcs)
    p,_ = ghost(p,x,bc_type = bc_type,bcs = bcs)
    u,x = ghost(u,x,bc_type = bc_type,bcs = bcs)
    #
    if rhs_forcing is None: 
        rhs_forcing = lambda u,x,t,dt,p: np.zeros(x.size)
    #
    if method == 'EF': #Euler
        if Full_Operator is None:
            Full_Operator = RHS_mat_adjoint(lambda_,u,x,t,dt,operators = operators)
        #
        A = np.eye(lambdas_.size) + dt * Full_Operator
        lambda_ = np.dot(Full_Operator,lambda_) + dt * rhs_forcing(lambda_,u,x,t,dt)
    #
    #
    if method == 'RK4':
        if Full_Operator is None:
            f = lambda lambda_,t,p: np.dot( RHS_mat_adjoint(lambda_,p[0],p[1],t,p[2],operators = p[3]) ,lambda_) + rhs_forcing(lambda_,p[0],p[1],t,p[2],p[4])
            rhs = RK4(f, lambda_, t, dt,p=[u,x,dt,operators,p])
        else: 
            f = lambda lambda_,t,p: np.dot(Full_Operator,lambda_) + dt * rhs_forcing(lambda_,p[0],p[1],t,p[2],p[3])
            rhs = RK4(f, lambda_, t, dt,p=[u,x,dt,p])
        #
        lambda_ = lambda_ + rhs
    lambda_,_ = deghost(lambda_,x,n_g = n_g)
    p,_ = deghost(p,x,n_g = n_g)
    u,x = deghost(u,x,n_g=n_g)
    #
    if return_rhs is True:
        lambda_ = (lambda_,rhs)
    if return_operator is True:
        if return_rhs is True:
            lambda_ = lambda_ + (Full_Operator,)
        else:
            lambda_ = (lambda_,Full_Operator)
        #
    #
    #
    return lambda_


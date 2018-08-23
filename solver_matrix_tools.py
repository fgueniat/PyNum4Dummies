import numpy as np
import weave
from operators_matrix_tools import *

'''
library for solving burgers and 1d pde.
contains 
an integrator (integration)
discretization of:
space derivative:
    LUD
    upwind
second space derivative
    central_scheme
cfl (compute and print the cfl number)

Some operators already discretized are imported from operators_matrix_tools.

'''

param_n_ghost = 8



acceleration_weave = True

##############################################################################################
##############################################################################################
############################## INTEGRATION PART ##############################################
##############################################################################################
##############################################################################################





def integration(u, x, t, dt, operators = None, rhs_forcing =None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False, full_operator = None, return_operator = False):
    '''
    bind to integration_forward_mat
    '''
    return integration_forward_mat(u,x,t,dt, operators = operators, rhs_forcing = rhs_forcing, method = method, bc_type = bc_type, bcs = bcs,n_g = n_g, return_rhs = return_rhs, full_operator = full_operator, return_operator = return_operator)

def integration_forward_mat(u, x, t, dt, operators = None, rhs_forcing = None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False,full_operator=None, return_operator = False):
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
    if rhs_forcing is None: 
        rhs_forcing = lambda u,x,t,dt: np.zeros(x.size)    
    #
    if method == 'EF': #Euler
        if full_operator is None:
            full_operator = RHS_mat(u,x,t,dt,operators = operators)
        #
        A =  dt * full_operator
        u = u + full_operator.dot(u) + dt * rhs_forcing(u,x,t,dt)
    #
    if method == 'RK4':
        if full_operator is None:
            f = lambda u,t,p: np.dot( RHS_mat(u,p[0],t,p[1],operators = p[2]) ,u) + rhs_forcing(u,p[0],t,p[1])
            rhs = RK4(f, u, t, dt,p=[x,dt,operators]) #+ rhs_forcing(u,x,t,dt) 
        else: 
            f = lambda u,t,p: full_operator.dot(u) +  rhs_forcing(u,p[0],t,p[1])
            rhs = RK4(f, u, t, dt,p=[x,dt]) #+ rhs_forcing(u,x,t,dt)
        #
        u = u+rhs
    u,x = deghost(u,x,n_g = n_g)
    #
    if return_rhs is True:
        u = (u,rhs)
    if return_operator is True:
        if return_rhs is True:
            u = u + (full_operator,)
        else:
            u = (u,full_operator)
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












##############################################################################################
##############################################################################################
############################## COMMON OPERATORS ##############################################
##############################################################################################
##############################################################################################








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
    nx = u.size
    if acceleration_weave is False:
        A = np.zeros((nx,nx))
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
    else: # use weave
        A = np.zeros((nx,nx))
        if np.isscalar(a):
            ap = np.ones(u.shape) * np.maximum(-a,0)
            am = np.ones(u.shape) * np.minimum(-a,0)
        else:
            ap = np.maximum(-a,0)
            am = np.minimum(-a,0)        
        dx = np.zeros(nx)
        code = '''            
            /* dx */
            for (int i=2;i<nx-2;i++){
                dx(i) = ( x(i+1) - x(i-1) )/ 2. ;
                }
            dx(1) = dx(2) ;
            dx(0) = dx(1) ;
            dx(nx-2) = dx(nx-3) ;
            dx(nx-1) = dx(nx-2) ;

            /* diag */
            for (int i=0;i<nx;i++){
                A(i,i) = ap(i) * 3. / ( 2. * dx(i) ) - am(i) * 3. / ( 2. * dx(i) ) ;
                }

            /* offset 1 */
            for (int i=1;i<nx;i++){
                /* diag -1 */
                A(i,i-1) = - ap(i) * 4. / ( 2.*dx(i) ) ;
                /* diag +1 */
                A(i-1,i) =  am(i-1) * 4. / ( 2.*dx(i-1) ) ;
                }

            /* offset 2 */
            for (int i=2;i<nx;i++){
                /* diag -2 */
                A(i,i-2) = ap(i) / ( 2.*dx(i) ) ;
                /* diag +2 */
                A(i-2,i) =  -am(i-2) / ( 2.*dx(i-2) ) ;
                }
            '''
        weave.inline(code,['A','nx','am','ap','dx','x'],type_converters=weave.converters.blitz)
        #A = -LUD_mat(a,u,x)
    return -A

def diffusion_mat(u,x):
    '''
    discretization of $ \frac{\partial^2 u} \{partial x^2} $ 
    central difference scheme scheme
    u is the quantity to be discretized
    x is the space
    '''
    nx = u.size
    d2udx2 = np.zeros(nx)
    A = np.zeros((nx,nx))
    if acceleration_weave is False: 
        dx = (x[2:] - x[:-2])/2.
        dx = np.r_[dx[0],dx,dx[-1]]
        v = np.ones(u.shape) / (2. * dx**2)
        #ui-1
        A += np.diag(v[1:],-1) 
        #ui
        A += -2.*np.diag(v) 
        #ui+1
        A += np.diag(v[1:], 1)
    else:
        A = np.zeros((nx,nx))
        dx = np.zeros(nx)
        code = '''            
            /* dx */
            for (int i=1;i<nx-1;i++){
                dx(i) = ( x(i+1) - x(i-1) )/ 2. ;
                }
            dx(0) = dx(1) ;
            dx(nx-1) = dx(nx-2) ;

            /* diag */
            for (int i=0;i<nx;i++){
                A(i,i) = -1. / ( dx(i) * dx(i) ) ;
                }

            /* offset 1 */
            for (int i=1;i<nx;i++){
                /* diag -1 */
                A(i,i-1) = 1. / ( 2.*dx(i) * dx(i) ) ;
                /* diag +1 */
                A(i-1,i) =  1. / ( 2.*dx(i-1) * dx(i-1) ) ;
                }

            '''
        weave.inline(code,['A','nx','dx','x'],type_converters=weave.converters.blitz)
        #A = diffusion_mat(u,x,False)

    return A



def dxxx_mat(u,x):
    '''
    discretization of $ \frac{\partial^3 u} \{partial x^2} $ 
    central difference scheme scheme:  $ \frac{\partial^3 u_i} \{partial x^2}  \approx \frac {u_{i+2} - u_{i-2} +2 u_{i-1} - 2u_{i+1}}{2\Delta x^3}
    u is the quantity to be discretized
    x is the space
    '''
    nx = u.size
    d2udx2 = np.zeros(nx)
    A = np.zeros((nx,nx))
    dx = (x[2:] - x[:-2])/2.
    dx = np.r_[dx[0],dx,dx[-1]]
    v = np.ones(u.shape) / (2. * dx**3)
    #ui-2
    A += - np.diag(v[1:-1],-2) 
    #ui-1
    A += 2.* np.diag(v[1:],-1) 
    #ui
    #ui+1
    A += -2.* np.diag(v[1:], 1)
    #ui+2
    A += np.diag(v[1:-1], 2)

    return -A





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




##############################################################################################
##############################################################################################
################################ ADJOINT PART ################################################
##############################################################################################
##############################################################################################

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

        



def integration_backward_mat(lambda_,u, x, t, dt, p_forcing=None, operators = None, rhs_forcing = None, method = 'RK4', bc_type = 'periodic', bcs = None, n_g = param_n_ghost, return_rhs = False,full_operator=None, return_operator = False,is_smooth = False):
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
    if operators is None:#graceful exit 
        print('no problem provided')
        return -1
    #
    lambda_,_ = ghost(lambda_,x,bc_type = bc_type,bcs = bcs)
    if p_forcing is not None :p_forcing,_ = ghost(p_forcing,x,bc_type = bc_type,bcs = bcs)
    u,x = ghost(u,x,bc_type = bc_type,bcs = bcs)
    #
    if rhs_forcing is None : 
        rhs_forcing = lambda u,x,t,dt,p: np.zeros(x.size)
    #
    if method == 'EF': #Euler. Not working ?
        if full_operator is None :
            full_operator = RHS_mat_adjoint(lambda_,u,x,t,dt,operators = operators)
        #
        A = np.eye(lambdas_.size) + dt * full_operator
        lambda_ = np.dot(full_operator,lambda_) + dt * rhs_forcing(lambda_,u,x,t,dt)
    #
    #
    if method == 'RK4':
        if full_operator is None : # construction of the function
            f = lambda lambda_,t,p: np.dot( RHS_mat_adjoint(lambda_,p[0],p[1],t,p[2],operators = p[3]) ,lambda_) +  rhs_forcing(lambda_,p[0],p[1],t,p[2],p[4])
            rhs = RK4(f, lambda_, t, dt,p=[u,x,dt,operators,p_forcing])# - rhs_forcing(lambda_,u,x,t,dt,p_forcing)/4.

        else : 
            f = lambda lambda_,t,p: np.dot(full_operator,lambda_) +  rhs_forcing(lambda_,p[0],p[1],t,p[2],p[3])
            rhs = RK4(f, lambda_, t, dt,p=[u,x,dt,p_forcing]) #- rhs_forcing(lambda_,u,x,t,dt,p_forcing)/4.
        #
        lambda_ = lambda_ + rhs
    if is_smooth is True : lambda_ = smooth(lambda_) # backward instabilities
    lambda_,_ = deghost(lambda_,x,n_g = n_g)
    if p_forcing is not None: p,_ = deghost(p_forcing,x,n_g = n_g)
    u,x = deghost(u,x,n_g=n_g)
    #
    if return_rhs is True :
        lambda_ = (lambda_,rhs)
    if return_operator is True :
        if return_rhs is True :
            lambda_ = lambda_ + (full_operator,)
        else :
            lambda_ = (lambda_,full_operator)
        #
    #
    #
    return lambda_


def lambda2mu(lambda0,dug, dupF):
    '''
    relationship between lambda and mu:
    $( \mu^t \partial_u g - lambda^T\partial_{\dot{u}}F)\\big|0 = 0 $
    To be done.
    Usually, \partial_{\dot{u}}F = 1, and \partial_u g = 1, hence mu = lambda(t=0)
    '''
    pass

def gradient_q(u0,U,lambdas,data,x,time,q,dqj,dqf,dqg,mu,desample=1) :
    '''
    Final integration for the gradient:
    $ D_q = \int_0-^T \partial_q j  + \lambda^T \partial_q F dt + \mu^T \partial_q g $
    
    dqj: list of n_q functions that represent $\partial_q j$. Functions have to accept the arguments (U,data,x,(i_t,t),q)
    dqf: list of n_q functions that represent $\partial_q F$. Functions have to accept the arguments (U,x,(i_t,t),q)
    dqg: list of n_q functions that represent $\partial_q g$. Functions have to accept the arguments (u0,q)
    mu can be computed by calling lambda2mu
    '''
    #
    n_q = len(q)
    n_t = len(time)
    #
    DJ = np.zeros(n_q)
    #
    #########################################################################
    # part $\int_0-^T \partial_q j  + \lambda^T \partial_q F dt$
    for i_t,t in enumerate(time) :
        if desample > 1 :
            is_sampled = False
            if i_t%desample == 0 : is_sampled = True
            if i_t == n_t-1 : is_sampled = True
        else : 
            is_sampled = True
        #
        if is_sampled is True :
            if i_t == n_t-1 :
                dt = (time[-1] - time[-2])
            elif i_t == 0 :
                dt = (time[1] - time[0])
            else :
                dt = 0.5 * (time[i_t+1] - time[i_t-1])
            #
            for i_q in range(n_q) :
                delta_dj =  dt * dqj[i_q]( U, data, x, (i_t,t), q )
                # the sum accelerate if dqf is a 0. instead of an operator
                delta_dj +=  dt * np.sum( 
                                    np.dot( 
                                    lambdas[i_t].T , 
                                    dqf[i_q]( U, x, (i_t,t), q ) 
                                    )
                                    )
                #
                DJ[i_q] += delta_dj
                #
            #
        #
    #
    if desample>1 : # interpolate
        from scipy.interpolate import interp1d as interp
        x,y = [],[]
        for i_q in range(n_q) :
            if np.abs(DJ[i_q])>1.e-10 :
                x.append(i_q)
                y.append(DJ[i_q])
        f = interp(x,y,kind = 'cubic')
        DJ = f(np.arange(n_q))
    #
    #########################################################################
    # part $\mu^T \partial_q g$
    for i_q in range(n_q) :
        dqg_i = dqg[i_q](u0,q)
        DJ[i_q] += np.dot(mu.T,dqg_i)
    return DJ




##############################################################################################
##############################################################################################
################################## SOME TOOLS ################################################
##############################################################################################
##############################################################################################

def smooth(x,n_s = 1):
    '''
    backward diffusion is unstable:
    smoothing helps by removing the unstabilities
    '''
    x_s = x.copy()
    x_s [n_s:-n_s] = (x[:-2*n_s] + x[n_s:-n_s] + x[2*n_s:])/3.
    return x_s

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

















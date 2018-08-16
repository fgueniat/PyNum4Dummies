import solver_matrix_tools as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as bench
from perlin_noise import perlin

'''
This script shows how to use the solver library, in the context of optimal control.
It means that the user has an objective.

1/ the user defines the physics (see script_pde.py)
2/ the user defines some of the adjoint operators
    i/ the adjoint equation is solved
    ii/ the gradient is computed
    iii/ goes to i/

'''

##################################################
##################################################
# parameters
##################################################
##################################################

# parameters for the minimization
n_adj_max = 10 # number of max iteration
eps_cost = 1.e-3 # stopping criterion when the cost does not decrease
update_,old_update_,eps_dq = 1.,1.,1.e-5 # stopping criterion when the the parameters do not change much
eps_delta_dq = 1.e-4  # stopping criterion when the the parameters do not change much
step_has_decreased = False # stoping criterion
islearning = True # learning a stepsize
gamma = 0.9 # decrease of the step at each iteration
gamma_step = 20. # increase of the step if the direction is good
stepsize_minimization = gamma_step / gamma 

# optimization
desample = 50

# type of integration
integration_method = 'RK4'

# physics
c_0 = 7.5 # advection speed
nu = 0.65 # dissipation
rho = .0 
is_adv,is_NL_adv,is_diff,is_dxxx = True, True, False,False
if is_diff is True : print('INFOS: backward integration is unstable')

# objectives
type_of_cost = ['T','t'] #T for final time, t for continuous time
objective_type = 't'
#space
n_x = 250 # number of space points
xmin,xmax = -12.,35. # boundaries of the space domain
x = np.linspace(xmin,xmax,n_x) # space domain
dx = x[2]-x[1] # space increment

# time
t0, tmax = 0.,30., # boundaries of the time domain
dt = 0.001 # time increment
n_it = int(tmax/dt) # number of time points
time = t0 + dt*np.arange(n_it) # time domain


# Initial Conditions
s_dev2, mu_offset = .3, -10.
u_background = 1.2
u0_init  = np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
s_dev2, mu_offset = .5, -7.
u0_init  += np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
u0_init += u_background

u_obj = 1.0 * np.ones(x.shape)
u_obj_final = 1.+ 0.5*np.sin(.5*x)*np.exp(-( (x-10)**2 )/50)

U_precomputed = None # u will be pre computed in the loop

# initial forcing
n_q = n_it # number of parameters
q = np.zeros(n_q) # initialization of the parameters
dq = np.zeros(n_q) # initialization of the gradient
old_dq = np.zeros(n_q) 

save_q = []
save_dq = []

old_cost = 1.e6
cost = 1.e5


# Boundary conditions
bc_type = 'dirichlet'
bcs = [u_background,u_background]
u_obj_ghost,_ = st.ghost(u_obj,x,bc_type = bc_type,bcs = bcs)
u_obj_final_ghost,_ = st.ghost(u_obj_final,x,bc_type = bc_type,bcs = bcs)

bc_noise = u_background + 1.*perlin(2.*time,seed=1) #upflow variation of traffic

inflow_noise =  4.*perlin(3.*time,seed=10) # mimic traffic comming from an intersection

# prints
verbose_computations = True
verbose_minimization = True
show_results_percentage = 33
ind_plot_start,ind_plot_end,n_plot_skip = 0,n_it,int(n_it/100)


############################################
##################################################
# define the equation:
##################################################
##################################################

### Physics
#

def f_bc(i_t):
    '''
    update boundary conditions
    '''
    #return u_background + 0.15*np.sin(i_t * dt * 2.*2.*np.pi) * np.sin(i_t * dt * .2*2.*np.pi)  
    return bc_noise[i_t]

def rhs(u,x,t,dt):
    '''
    right hand side
    it mimics an afflux of traffic (incoming from an intersection) and a deflux of traffic (again from an intersection)
    '''
    v = np.zeros(u.size)
    #
    s_dev2_rhs, mu_offset_rhs = .2, 5.
    #v  +=   0.6 * \
    #        (1. + \
    #            np.sin(1.2*t*2.*np.pi) * \
    #            np.cos(1.4*t*2.*np.pi) * \
    #( 1.+np.exp(- (t-4 )**2 / 20. ) +np.exp(- (t-10 )**2 / 20. )     )) * \
    #    np.exp(- (x-mu_offset_rhs -2 )**2 / (2. * s_dev2_rhs) )  
    v += inflow_noise[int(t/dt)-1] * np.exp(- (x-mu_offset_rhs -2 )**2 / (2. * s_dev2_rhs) )  
    #s_dev2_rhs, mu_offset_rhs = .24, 13.
    #v  += - 0.6 * (1. + np.cos(0.8*t*2.*np.pi)  ) * np.exp(- (x-mu_offset_rhs -2 )**2 / (2. * s_dev2_rhs) ) 
    return v

def rhs_adjoint(lambda_,u,x,t,dt,p):
    '''
    this adjoint operator O is the RHS part of the adjoint equation, the linearization of the cost w.r.t. u:
    O(lambda,u,x,t,p) = dj/du
    '''
    #return np.zeros(lambda_.size)
    rhs = np.zeros(lambda_.size)
    #
    if objective_type == 'T':
        #||u(T) - u_obj|| ^2
        if t >= tmax +  2.*dt: #dt is negative 
            rhs += 2. * (u-u_obj_final_ghost)#
    if objective_type == 't':
        #||u(t) - u_obj|| ^2
        rhs += 2. * (u-u_obj_ghost)#
    return rhs


### model + Adjoint
#  Construction of individual operators:
s_dev2_control, mu_offset_control = .1, 0.
def operator_forcing_full(u,x,t,dt,p):
    '''
    operator for control
    '''
    v  = p*np.exp(- (x-mu_offset_control)**2 / (2. * s_dev2_control) )
    return v

def dqf(u,x,t,q,i):
    '''
    derivative of f, the forcing, w.r.t. control parameters
    '''
    if t[0] == i:
        return np.exp(- (x-mu_offset_control)**2 / (  2. * s_dev2_control) )
    else:
        return 0. #faster
        #return np.zeros(x.shape)


operator_advection              = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c_0)
operator_NL_advection           = lambda u,x,t,dt: st.operator_NL_advection(u,x,t,dt,p=1.)
operator_diffusion              = lambda u,x,t,dt: st.operator_diffusion(u,x,t,dt,p=nu)    

operator_advection_adjoint      = lambda lambda_,u,x,t,dt: st.operator_advection_adjoint(lambda_,u,x,t,dt,p=c_0)
operator_NL_advection_adjoint   = lambda lambda_,u,x,t,dt: st.operator_NL_advection_adjoint(lambda_,u,x,t,dt,p=1.)
operator_diffusion_adjoint      = lambda lambda_,u,x,t,dt: st.operator_diffusion_adjoint(lambda_,u,x,t,dt,p=nu)  

#    
operators_without_control = []
operators_adjoint = []
if is_adv is True :
    operators_without_control += [operator_advection]
    operators_adjoint +=  [operator_advection_adjoint]
if is_NL_adv is True :
    operators_without_control += [operator_NL_advection]
    operators_adjoint +=  [operator_NL_advection_adjoint]
if is_diff is True :
    operators_without_control += [operator_diffusion]
    operators_adjoint +=  [operator_diffusion_adjoint]








##################################################
##################################################
# Functions for BFGS:
##################################################
##################################################
def physics(q,verbose = False):
    '''
    # Compute the solution of the model.
    '''
    # Initialize solution
    U,t_sampled = [],[]
    u = u0_init.copy()
    # Loop
    for i_t in xrange(n_it):
        # redefine the forcing
        operator_forcing = lambda u,x,t,dt: operator_forcing_full(u,x,t,dt,q[i_t]) + rhs(u,x,t,dt)
        operators = operators_without_control
        # boundary conditions: can be time dependant !
        bcs[0] = f_bc(i_t)         
        u = st.integration(u,x,time[i_t],dt,operators,method=integration_method,bc_type = bc_type, bcs = bcs, return_rhs = False,rhs_forcing = operator_forcing)
        #
        if verbose is True:
            if i_t%int(n_it * show_results_percentage/100.) == 0:
                s = 'computations U:' + '%.0f' % (int(100.*i_t/n_it)) + '%'
                print(s)
            #
        #
        U.append(u.copy())
    return U

def adjoint(q,U=None,verbose = False):
    '''
    # Compute the adjoint states
    '''
    if U is None: U=physics(q)
    # initialize lambda
    lambda0 = np.zeros(n_x)
    # lambda
    lambdas = []
    n_save = 1
    lambda_ = lambda0.copy()
    # Loop
    for i_t in xrange(n_it-1,-1,-1):
        lambda_ = st.integration_backward_mat(lambda_,U[i_t],x,time[i_t],-dt,p_forcing = None, operators=operators_adjoint,rhs_forcing = rhs_adjoint,method=integration_method,bc_type = bc_type, bcs = [0.,0.],return_rhs = False)
        ###############
        ###############
        if verbose is True:
            if i_t%int(n_it * show_results_percentage /100.) == 0:
                s = 'computations adjoint:' + '%.0f' % (int(100.*i_t/n_it)) + '%'
                print(s)
            #
        #
        if i_t%n_save == 0:#save only one solution every n_save 
            lambdas.append(lambda_.copy())
        #
    # integration was backward: reordering terms
    lambdas = lambdas[::-1]
    return lambdas

def cost_function(q,U=None):
    '''
    Objective function to minimize
    '''
    if U is None : U=physics(q)
    if objective_type == 'T':
        return np.linalg.norm(U[-1] - u_obj)
    if objective_type == 't':
        return np.mean([np.linalg.norm(u - u_obj) for u in U])

def gradient(q,U=None,lambdas=None,desample=desample):
    '''
    compute the gradient of the cost_function w.r.t. the parameters.
    '''
    if U is None : U = physics(q)
    if lambdas is None : lambdas = adjoint(q,U)
    #
    # mu    
    #relationship between lambda and mu:
    #$( \mu^t \partial_u g - lambda^T\partial_{\dot{u}}F)\\big|0 = 0 $
    #
    mu = lambdas[0].copy() 
    #
    # final integration to compute the gradient
    ############# TODO :: PUT EVERYTHING HERE IN operators_matrix_tools !! ####################
    # a few definitions:
    # dqj: partial derivative of the cost w.r.t. the parameters
    null_dqj = lambda u,d,x,t,q: 0.
    # dqf : partial derivative of the model w.r.t. the parameters
    null_dqf = lambda u,x,t,q: np.zeros(x.size) # model
     # dqg  : partial derivative of the initial conditions w.r.t. the parameters
    null_dqg = lambda u,q: np.zeros(u.shape) 
    #
    def dqji(q,i):
        '''
        partial derivation of the cost w.r.t. the ith component of the parameters
        '''
        dqj = 2.*rho * q[i]
        return dqj
    #        
    operator_dqj = [( lambda i: lambda u,x,data,t,q: dqji(q,i) )(i) for i in range(n_q)]
    operator_dqf = [(lambda i: lambda u,x,t,q: dqf(u,x,t,q,i) )(i) for i in range(n_q)] 
    operator_dqg = [null_dqg for i in range(n_q)]
    #
    DJ = st.gradient_q(u0_init,U,lambdas,None,x,time,q,operator_dqj,operator_dqf,operator_dqg,mu,desample=desample)
    #
    return DJ

##################################################
##################################################
# Start the optimization:
##################################################
##################################################
time_offset = bench.time()
for i_adjoint in range(n_adj_max):
    if np.isnan(q).any():
        print('q has a NaN')
        break
    #
    # to stabilize the minimization
    stepsize_minimization = gamma * stepsize_minimization 
    #
    # print
    if verbose_minimization is True : 
        print('####################################################')
        print('iteration ' + str(1+i_adjoint) + ' over ' + str(n_adj_max) )
    #########################################################
    ## Stopping Criteria
    #
    if step_has_decreased is False:
        if i_adjoint>1:
            if np.linalg.norm(update_)/np.linalg.norm(q) <eps_dq:
                print('gradient is small: break')
                break
        #
        if i_adjoint>2:
            if np.linalg.norm(update_ - old_update_)/np.linalg.norm(update_) < eps_delta_dq:
                print('gradient update is small: break')
                break
            #
        #
        if i_adjoint>5:
            if np.abs( cost - old_cost )/np.abs(cost)<eps_cost:
                print('cost update is small: break')
                break
            #
        #
    #
    ###########################33
    # Computing the solution
    if U_precomputed is not None : U = physics(q)
    else : U = U_precomputed
    cost = cost_function(q,U)
    #
    ############################
    # Computing the adjoint state
    if verbose_minimization is True : print('computations of the adjoint state in progress')
    lambdas = adjoint(q,U)
    #
    ################################
    # Computing the gradient
    if verbose_minimization is True : print('computations of the gradient in progress')
    dq = gradient(q,U,lambdas) 
    #
    ################################
    # Update q and the data
    new_q = q + stepsize_minimization * dq
    U_precomputed = physics(new_q)
    new_cost = cost_function(new_q,U_precomputed)
    ################################
    # Adjusting the step size:
    # spirit of BFGS with a 1 step line search
    if islearning is True :
        step_has_decreased = False
        if verbose_minimization is True : print('adjusting the step length in progress')
        if new_cost > cost : # stepsize is too large, reset or the stepsize
            if stepsize_minimization>1.:
                stepsize_minimization = 1.
            else :
                stepsize_minimization = stepsize_minimization / gamma_step
            new_q = q # no update of q
            U_precomputed = U # no update of U
            new_cost = cost # no update the cost
            step_has_decreased = True
        else : # stepsize can *possibly* be increased
            test_q = q + gamma_step * stepsize_minimization * dq
            U_test = physics(test_q)
            test_cost = cost_function(test_q,U_test)
            if test_cost < new_cost : # if it is better, stepsize can be increased
                stepsize_minimization = gamma_step * stepsize_minimization #update of the stepsize
                U_precomputed = [u.copy() for u in U_test] # update U
                new_q = q + stepsize_minimization * dq # update q
                new_cost = test_cost # update the cost
            #
        #
    #################################
    # Savings and stopping criteria
    old_cost = cost +0. #for stopping criteria
    cost = new_cost +0.
    update_ = new_q - q
    # 
    q = new_q # update q
    save_q.append(q.copy())
    save_dq.append(stepsize_minimization * dq)
    if i_adjoint>0:
        old_update_ = save_dq[-2].copy()
    #
    if islearning is True : 
        if verbose_minimization is True:
            print('learning factor = ' + "%.2f" %stepsize_minimization )    
    #  
    if verbose_minimization is True :
        print('current cost function f(q) = ' + "%.2f" %(new_cost) )
        if i_adjoint>1:
            print('last update in cost %.4f'%(cost-old_cost) )
        #
    #
time_minimization = bench.time()  - time_offset
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('Optimization is (hopefully) a sucess')
print('time needed: %.2f' %time_minimization)
##################################################################
#################################################################
# Plots

U = physics(q)
U_without_control = physics(np.zeros(n_q))
lambdas = adjoint(np.zeros(n_q),U_without_control)

# Plot the adjoint + solution + forcage
if 1 :
    adj_fact = 1
    #
    def update_line(num, lambdas, U,U_wc,x,line1,line2,line3,line4,line5,line6,title):
        '''
        update the plot
        '''
        xs1 = c_0 * (tmax-time[num] )
        xs = ((xs1)%10.) - 5.
        ind = np.argmin(np.abs(x - xs))
        title_text = 'time: %.2f' %time[num]
        title_text += ' over %.2f' %time[-1]
        u_obj_transp = u_obj.copy()
        line1.set_data(x,adj_fact*lambdas[num]+.01)
        line2.set_data(x,U[num])
        line3.set_data(x,U_wc[num])
        line4.set_data(x,operator_forcing_full(None,x,None,dt,q[num]))
        line5.set_data(x,rhs(U[num],x,time[num],dt))
        line6.set_data(x,u_obj)
        title.set_text(title_text)
        #
        return line1,line2,line3,line4,line5,line6,title
    #
    #
    fig, ax = plt.subplots()
    #
    line1, = ax.plot([], [], 'r-')
    line2, = ax.plot([], [], 'g-')
    line3, = ax.plot([], [], 'g--')
    line4, = ax.plot([], [], 'b-')
    line5, = ax.plot([], [], 'b--')
    line6, = ax.plot([], [], 'k-')
    title_time = ax.text(0.5,.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center")
    plt.xlim(x.min(),x.max())
    umax = np.max([
        np.max(np.abs(np.array(U))), 
        np.max( np.abs(np.array(lambdas))), 
        np.max(np.abs(q))
        ])
    umin = -umax
    umin,umax = -4.,2.
    plt.ylim(umin,umax)
    plt.title('adjoint, burgers')
    plt.xlabel('x')
    plt.ylabel('$\lambda$,$u$')
    plt.legend([str(adj_fact) + '*adjoint','u','controlled u','control','rhs','obj'])
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(lambdas,U,U_without_control,x,line1,line2,line3,line4,line5,line6,title_time),
                                       interval=150, blit=False)
    #
    line_ani.running=True
    # This function will toggle pause on mouse click events
    def on_click(event):
        if line_ani.running == True :
            line_ani.event_source.stop()
            line_ani.running = False
        else:
            line_ani.event_source.start()    
            line_ani.running = True
    #
    # Here we tell matplotlib to call on_click if the mouse is pressed
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()





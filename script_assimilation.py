import solver_matrix_tools as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from perlin_noise import perlin

'''

This script shows how to use the solver library, in the context of data assimilation.
It means that the user have some data at hand.
This data is used either for:
    a/ identify some parameters of the model
    b/ identify initial conditions for the model that allow to reproduce as good as possible the data
    c/ both a/ and b/

1/ the user defines the physics (see script_pde.py)
2/ the user defines some of the adjoint operators
3/ 
    i/ parameters are updated
    ii/ the model is integrated with the current parameters
    iii/ the adjoint equation is solved
    iv/ the gradient is computed
    v/ goes to i/

'''


##################################################
##################################################
# parameters
##################################################
##################################################

# Data Assimilation objective:
objectives = ['CI and MODEL','MODEL','CI']
objective = objectives[0]
is_adv,is_nl_adv,is_diff,is_dxxx = True,True,True,False
smooth_step = 2
use_noise = True
if is_diff is True : print('INFOS: backward integration is unstable with viscosity')
# parameters for the minimization
n_adj_max = 40 # number of updates
eps_DJ, eps_delta_DJ = 1.e-7,1.e-6 #when to stop
gamma,gamma_step = .9,5. #lowering or increasing the steps
max_c, max_u = 10.,10., # control for stability of the model
min_nu,max_nu = .1, 1. # idem
stepsize_minimization = 1. # q = q + step * dq


# type of integration
integration_method = 'RK4'

# physics
q_physics= [7.2 , 0.65] # this is the initial parameters for c and nu

#
if objective == 'CI': 
    q_data = q_physics # parameters of data are the parametres of the model
else :
    q_data = [6.5,0.45] # we have a different set of initial parameters for the model and for the data

c_0,nu = q_physics[0],q_physics[1]
c_data,nu_data = q_data[0],q_data[1]

#space
n_x = 200
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
t0,tmax = 0.,1.5
dt = 0.001
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)

# Initial Conditions
s_dev2, mu_offset = .3, -1.
u_background = 0.
u0_init  = np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
u0_init += u_background

# Data
if objective == 'CI' or objective == 'CI and MODEL': #the IC are different for the model and the data
    u_background_data = 0.
    s_dev2_data, mu_offset_data = 3., 1.
    if use_noise is True :
        u_data  = u_background + perlin(np.linspace(0,5,n_x),seed=10) * np.exp(- (x)**2 / (2. * s_dev2_data) )
    else :
        u_data  = np.exp(- (x-mu_offset_data)**2 / (2. * s_dev2_data) ) / np.sqrt(2. * np.pi * s_dev2_data)
        u_data += u_background_data
else : # if CI are similar
    u_data = u0_init.copy()


# parameters
if objective == 'MODEL':
    n_q = 2
    q = q_physics
if objective == 'CI':
    n_q = n_x
    q = u0_init
if objective == 'CI and MODEL':
    n_q = n_x + 2
    q = np.r_[u0_init,q_physics]

# Boundary conditions
bc_type = 'periodic'
bcs = None

# plot parameters:
ind_plot_start,ind_plot_end,n_plot_skip = 0,n_it,int(n_it/100)
verbose_data = False
verbose_minimization = True

def limiters(u,max_,min_ = None):
    '''
        force the solution to remain in a feasible domain
    '''
    if u > max_:
        u = max_
    if min_ is None:
        if u < -max_:
            u = -max_
    else :
        if u<min_:
            u = min_
    return u

##################################################
##################################################
# define the equation:
##################################################
##################################################

### Physics
#
def rhs(u,x,t,dt):
    return np.zeros(u.size)

def rhs_adjoint(lambda_,u,x,t,dt,p):
    '''
    this adjoint operator O is the RHS part of the adjoint equation, the linearization of the cost w.r.t. u:
    O(lambda,u,x,t,p) = dj/du
    '''
    #return np.zeros(lambda_.size)
    return 2. * (u-p) # j = ||u-y|| ^2


#  Construction of the operators:

def construct_operator(c,nu,p=1,is_adv = is_adv, is_nl_adv = is_nl_adv, is_diff = is_diff, is_dxxx = is_dxxx):
    '''
    This function assembles the operators needed for the physics
    '''
    #  Construction of individual operators:
    operator_advection         = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c)
    operator_NL_advection      = lambda u,x,t,dt: st.operator_NL_advection(u,x,t,dt,p=1.)
    operator_diffusion         = lambda u,x,t,dt: st.operator_diffusion(u,x,t,dt,p=nu)        # 
    operator_dxxx              = lambda u,x,t,dt: st.operator_dxxx(u,x,t,dt,p=nu)  
    #  Assemble the individual operators:
    operators = []
    if is_adv is True :
        operators += [operator_advection]
    if is_nl_adv is True :
        operators += [operator_NL_advection]  
    if is_diff is True :
        operators += [operator_diffusion]
    if is_dxxx is True:
        operators += [operator_dxxx]
    return operators


def construct_operator_adjoint(c,nu,p=1,is_adv = is_adv, is_nl_adv = is_nl_adv, is_diff = is_diff, is_dxxx = is_dxxx):
    '''
    This function assembles the operators needed for computing the adjoint state
    '''
    #  Construction of the individual operators:
    operator_advection_adjoint      = lambda lambda_,u,x,t,dt: st.operator_advection_adjoint(lambda_,u,x,t,dt,p=c)
    operator_NL_advection_adjoint   = lambda lambda_,u,x,t,dt: st.operator_NL_advection_adjoint(lambda_,u,x,t,dt,p=1.)
    operator_diffusion_adjoint      = lambda lambda_,u,x,t,dt: st.operator_diffusion_adjoint(lambda_,u,x,t,dt,p=nu)  
    operator_dxxx_adjoint           = lambda lambda_,u,x,t,dt: st.operator_dxxx_adjoint(lambda_,u,x,t,dt,p=1.)
    #
    #  Assemble the individual operators:
    operators_adjoint = []
    if is_adv is True :
        operators_adjoint += [operator_advection_adjoint]
    if is_nl_adv is True :
        operators_adjoint += [operator_NL_advection_adjoint]
    if is_diff is True :
        operators_adjoint += [operator_diffusion_adjoint]
    if is_dxxx is True :
        operators_adjoint += [operator_dxxx_adjoint]
    return operators_adjoint

# storages
DJ = np.zeros(n_q)
gradients = []
u0s = []
lambdas_s = []

##################################################
##################################################
# Computations:
##################################################
##################################################


def physics(c_0,nu,u0):
    '''
    # Compute the solution of the model.
    '''
    ##################################################
    # define the equation for the model:
    ##################################################
    operators = construct_operator(c_0,nu)
    #
    #
    ##################################################
    # Compute the solution
    ##################################################
    # Initialize solution
    U,t_sampled = [],[]
    n_save = 1
    u = u0.copy()
    # Loop
    for i_t in xrange(n_it):
        u = st.integration(u,x,time[i_t],dt,operators,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False)
        #
        if verbose_data is True:
            if i_t%int(n_it/4.) == 0:
                s = 'computations U:' + '%.0f' % (4.*i_t/int(n_it/25)) + '%'
                print(s)
            #
        #
        if i_t%n_save == 0:#save only one solution every n_save 
            U.append(u.copy())
            t_sampled.append(time[i_t])
        #    return 
    return U,t_sampled


def adjoint(c_0,nu,u0,U_data,U=None):
    '''
    # Compute the adjoint states
    '''
    if objective == 'CI':
        q = u0.copy()
    if objective == 'MODEL':
        q = [c_0,nu]
    if objective == 'CI and MODEL' :
        q = np.r_[u0.copy(),np.array([c_0,nu])]
    if U is None:
        U,_ = physics(c_0,nu,u0)
    ########
    #  Construction of the operators:
    operators_adjoint = construct_operator_adjoint(c_0,nu)
     #
    lambda0 = np.zeros(n_x)
    # lambda
    if 1:
        lambdas = []
        n_save = 1
        lambda_ = lambda0.copy()
        # Loop
        for i_t in xrange(n_it-1,-1,-1):
            if is_diff is True : # smoothing for backward integration when viscosity
                if i_t % smooth_step == 0 : is_smooth = True
                else : is_smooth = False
            else : is_smooth = False
            lambda_ = st.integration_backward_mat(lambda_,U[i_t],x,time[i_t],-dt,U_data[i_t],operators=operators_adjoint,rhs_forcing = rhs_adjoint,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False,is_smooth = is_smooth )
            if verbose_data is True:
                if i_t%int(n_it/4.) == 0:
                    s = 'computations adjoint:' + '%.0f' % (4.*i_t/int(n_it/25)) + '%'
                    print(s)
                #
            #
            if i_t%n_save == 0:#save only one solution every n_save 
                lambdas.append(lambda_.copy())
            #
        #
    #
    # integration was backward: reordering terms
    lambdas = lambdas[::-1]
    return lambdas

def gradient(c_0,nu,u0,U_data,U = None, lambdas = None):
    '''
    compute the gradient of the cost_function w.r.t. the parameters.
    '''
    if objective == 'CI':
        q = u0.copy()
    if objective == 'MODEL':
        q = [c_0,nu]
    if objective == 'CI and MODEL' :
        q = np.r_[u0.copy(),np.array([c_0,nu])]
    #
    if U is None :
        U,_ = physics(c_0,nu,u0)

    if lambdas is None :
        lambdas = adjoint(c_0,nu,u0,U_data,U)
    # mu
    #relationship between lambda and mu:
    #$( \mu^t \partial_u g - lambda^T\partial_{\dot{u}}F)\\big|0 = 0 $
    #
    mu = lambdas[0].copy() 
    #
    # final integration to compute the gradient
    # a few definitions:
    # dqj:
    null_dqj = lambda u,d,x,t,q: 0.
    # dqf
    null_dqf = lambda u,x,t,q: np.zeros(x.size) # model
    def dqf_c (U,x,t,q):
        uu = U[t[0]].copy()
        uu,xx = st.ghost(uu,x)
        M = st.LUD_mat(-1.,uu,xx) # LHS
        uu = np.dot(M,uu)
        u,_ = st.deghost(uu,xx)
        return u
    #
    def dqf_nu (U,x,t,q):
        uu = U[t[0]].copy()
        uu,xx = st.ghost(uu,x)
        M = st.diffusion_mat(uu,xx) #LHS
        uu = np.dot(M,uu)
        u,_ = st.deghost(uu,xx)
        return u    
    # dqg
    null_dqg = lambda u,q: np.zeros(u.shape) 
    def ci_dqg(u,q,i):
        dqgi = np.zeros(u.size)
        if i < u.size:
            dqgi[i]=1.
        #
        return dqgi
    #
    #
    if objective == 'MODEL':
        operator_dqj = [null_dqj for i in range(n_q)]
        operator_dqf = [dqf_c,dqf_nu]
        operator_dqg = [null_dqg for i in range(n_q)]
    #
    if objective == 'CI':
        operator_dqj = [null_dqj for i in range(n_q)]
        operator_dqf = [null_dqf for i in range(n_q)] 
        operator_dqg = [( lambda i: lambda u,q: ci_dqg(u,q,i) )(i) for i in range(n_q)]
    #
    if objective == 'CI and MODEL':
        operator_dqj = [null_dqj for i in range(n_q)]
        operator_dqf = [null_dqf for i in range(n_x)] + [dqf_c,dqf_nu]
        operator_dqg = [( lambda i: lambda u,q: ci_dqg(u,q,i) )(i) for i in range(n_q)]
    #
    DJ = st.gradient_q(u0,U,lambdas,U_data,x,time,q,operator_dqj,operator_dqf,operator_dqg,mu)
    #
    return DJ


def update_q(q,dq,step):
    if objective == 'MODEL':
        return q + dq * step
    #
    if objective == 'CI':
        return q + dq * step
    #
    if objective == 'CI and MODEL':
        q_p1 = q.copy()
        q_p1[:-2] = q_p1[:-2] + step[0] * dq[:-2] 
        q_p1[-2:] = q[-2:] + step[1] * dq[-2:]
        return q_p1

def cost_function(U,U_data):
    if U is None: return 1.e20
    return dt * np.sum([np.linalg.norm(U[i] - U_data[i]) for i in xrange(n_it)]) 



def line_search(q,dq,step,U_data,U=None):
    step_base = gamma * step
    if objective == 'CI':
        c_0,nu = q_physics[0], q_physics[1]
        u0 = q.copy()
        du = dq.copy()
        dc,dnu =0.,0.
    if objective == 'MODEL':
        u0 = u0_init
        c_0,nu = q[0],q[1]
        dc,dnu = dq[0],dq[1]
        du = 0.
    if objective == 'CI and MODEL' :
        u0 = q[:-2]
        du = dq[:-2]
        c_0,nu = q[-2],q[-1]
        dc,dnu = dq[-2],dq[-1]
    if U is None : U,_ = physics(c_0,nu,u0)
    cost_0 = cost_function(U,U_data)
    cost_base = cost_0
    #
    n_ls = 3  
    next_step = step_base
    is_better = False
    for i_ls in range(n_ls):
        next_U,_ = physics(c_0 + next_step *dc, nu + next_step * dnu, u0 + next_step * du)
        next_cost = cost_function(next_U,U_data)
        if next_cost < cost_base:
            is_better = True
            U_base = [u.copy() for u in next_U]
            cost_base = next_cost
            step_base = next_step
            next_step = next_step * gamma_step
        else :
            next_step = next_step/3.
    # what happens if no points are better ?
    if is_better is False :
        print 'no better minimum than current point. Will diminish the step'
        return U, next_step
    else :        
        return U_base, step_base


######################## DATA
U_data,_ = physics(c_data,nu_data,u_data)

######################## MINIMIZATION
for i_adjoint in range(n_adj_max):
    #
    if i_adjoint > 1:
        if np.isnan(cost_function(U,U_data)):
            print ('something went wrong')
            break
    if i_adjoint>1:
        if np.linalg.norm(DJ)<eps_DJ:
            print('gradient is small: break')
            break
    if i_adjoint>2:
        if np.linalg.norm(gradients[-1] - gradients[-2]) < eps_delta_DJ:
            if do_computations is True :
                print('gradient update is small: break')
                break
    ## update model:
    # update all parameters with the gradient
    q = q + DJ * stepsize_minimization
    #
    # update model parameters
    if objective == 'CI':
        c_0,nu = q_physics[0],q_physics[1]
    else:
        q[-2] = limiters(q[-2],max_c)
        q[-1] = limiters(q[-1],max_nu,min_nu)
        c_0,nu = q[-2],q[-1]
    #
    # update initial conditions:
    if objective == 'CI' or objective == 'CI and MODEL':
        for i_x in range(n_x): 
            q[i_x] = limiters(q[i_x],max_u)
        u0 = q[:n_x]
    else:
        u0 = u0_init
    #
    ##################################################
    # Run the model
    ##################################################
    if i_adjoint == 0: 
        U,_ = physics(c_0,nu,u0) # otherwise send back from line search
    if i_adjoint == 0 :
        U_init = [u.copy() for u in U]
    ##################################################
    # Compute the gradient
    ##################################################
    # adjoint states
    lambdas = adjoint(c_0,nu,u0,U_data,U)
    # gradient
    DJ = gradient(c_0,nu,u0,U_data,U,lambdas)
    U,stepsize_minimization = line_search(q,DJ,stepsize_minimization,U_data,U)
    print stepsize_minimization
    ########################################################
    ################### PRINTS #############################
    if verbose_minimization is True:
        s = '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print(s)
        s  = 'iteration : ' + str(i_adjoint)
        print(s)
        #
        s  = 'Cost : '
        s += '%.4f' %(cost_function(U,U_data))
        print(s)
        if (objective=='CI') is False:
            s  = 'targets are:        '
            s += ' c: '
            s += '%.2f' % q_data[0]
            s += ' nu: '
            s += '%.2f' % q_data[1]
            print(s)
            s  = 'initial guesses are:'
            s += ' c: '
            s += '%.2f' % c_0
            s += ' nu: '
            s += '%.2f' % nu
            print(s)
            s  = 'Gradient DJ (model) is:'
            s += '( %.4f ,' % DJ[-2]
            s += ' %.4f )' % DJ[-1]
            #print(s)
            s  = 'Used Gradient DJ (model) is:'
            s += '( %.4f ,' % (stepsize_minimization * DJ[-2])
            s += ' %.4f )' % (stepsize_minimization * DJ[-1])
            print(s)
        if  (objective=='MODEL') is False:
            s  = 'ecart CI: '
            s += '%.4f' % (np.linalg.norm(u0 + DJ[:n_x] - u_data)/np.linalg.norm(u_data))
            print(s)
    #############################################################
    ############################################################
    # save
    #
    u0s.append(u0.copy())
    lambdas_s.append(lambdas[0].copy())
    gradients.append(DJ)
    #
#




##################################################
##################################################
# Plots
##################################################
##################################################
#
# Plot the solution
if 0:
    def update_line(num, U,x,line,verbose = True):
        '''
        update the plot
        '''
        line.set_data(x,U[num])
        if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
        return line,
    #
    fig, ax = plt.subplots()
    #
    line, = ax.plot([], [], 'r-')
    plt.title('burger')
    plt.xlim(x.min(),x.max())
    plt.ylim(u0.min()-1, u0.max()+1)
    plt.xlabel('x')
    plt.ylabel('u')
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(U,x,line),
                                       interval=100, blit=True)
    #
    plt.show()


#
# Plot the adjoint
if 0:
    def update_line_adjoint(num, lambdas,x,line,verbose = True):
        '''
        update the plot
        '''
        line.set_data(x,lambdas[num])
        if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
        return line,
    #
    fig2, ax2 = plt.subplots()
    #
    line, = ax2.plot([], [], 'r-')
    plt.xlim(x.min(),x.max())
    plt.ylim(lambdas[0].min()-1, lambdas[0].max()+1)
    plt.title('adjoint')
    plt.xlabel('x')
    plt.ylabel('$\lambda$')
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig2, update_line_adjoint, xrange(ind_plot_end-1,ind_plot_start,-n_plot_skip), fargs=(lambdas,x,line),
                                       interval=100, blit=True)
    #
    plt.show()

#
# Plot the adjoint + solution
if 1:

    #
    #
    fig3, ax3 = plt.subplots()
    #
    line21, = ax3.plot([], [], 'r-')
    line22, = ax3.plot([], [], 'k-')
    line23, = ax3.plot([], [], 'k--')
    line24, = ax3.plot([], [], 'g-')
    title_time = ax3.text(0.5,
            .95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax3.transAxes, ha="center")
    plt.xlim(x.min(),x.max())
    plt.ylim(lambdas[0].min()-1, lambdas[0].max()+1)
    plt.title('adjoint, burgers')
    plt.xlabel('x')
    plt.ylabel('$\lambda$,$u$')
    plt.legend(['adjoint','u','u_init','data'])
    #
    plt.ion()
    #
    def update_line(num, lambdas, U,U_init,V,x,line1,line2,line3,line4,title):
        '''
        update the plot
        '''
        title_text = 'time: %.2f' %(time[num])
        title_text += ' over %.2f' %(time[-1])
        line1.set_data(x,lambdas[num]+.01)
        line2.set_data(x,U[num])
        line3.set_data(x,U_init[num])
        line4.set_data(x,V[num])
        title.set_text(title_text)
        mmax,mmin = np.max([U[num],lambdas[num]]),np.min([U[num],lambdas[num]])
        ax3.set_ylim(mmin,mmax)
        return line1,line2,line3,line4,title,
    line_ani = animation.FuncAnimation(fig3, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(lambdas,U,U_init,U_data,x,line21,line22,line23,line24,title_time),
                                       interval=100, blit=False)
    #
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
    cid = fig3.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# Plot the adjoint + solution
if 0 :
    if objective != 'MODEL':
        #
        fig4, ax4 = plt.subplots()
        #
        line41, = ax4.plot([], [], 'r-')
        line42, = ax4.plot([], [], 'k-')
        title_time = ax4.text(0.5,
            .95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax4.transAxes, ha="center")
        plt.xlim(x.min(),x.max())
        ax4.set_ylim(lambdas[0].min(), lambdas[0].max())
        def update_line(num, lambdas,U,u0,x,line1,line2,title):
            '''
            update the plot
            '''
            title_text = 'iteration: %.0f' %num
            title_text += ' over %.0f' %n_adj_max
            line1.set_data(x,lambdas[num])
            line2.set_data(x,U[num]-u0)
            ax4.set_ylim((U[num]-u0).min(),(U[num]-u0).max())
            title.set_text(title_text)
            return line1,line2,title,
        #
        plt.title('adjoint, burgers')
        plt.xlabel('x')
        plt.ylabel('$\lambda$,$u$')
        plt.legend(['adjoint','$u_i-u_d$'])
        #
        plt.ion()
        #
        line_ani = animation.FuncAnimation(fig4, update_line, xrange(0,n_adj_max), fargs=(lambdas_s,u0s,u_data,x,line41,line42,title_time),
                                           interval=300, blit=False)
        #
        #
        plt.show()







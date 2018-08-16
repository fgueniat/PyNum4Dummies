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
is_adv,is_nl_adv,is_diff,is_dxxx = True,False,True,False
use_noise = True
if is_diff is True : print('INFOS: backward integration is unstable with viscosity')
# parameters for the minimization
n_adj_max = 55
eps_DJ = 1.e-7
eps_delta_DJ = 1.e-6

# type of integration
integration_method = 'RK4'

# physics
q_physics= [7.2 , 0.35]

#
if objective == 'CI':
    q_data = q_physics
else :
    q_data = [6.5,0.45]
c_0,nu = q_physics[0],q_physics[1]
c_data,nu_data = q_data[0],q_data[1]

#space
n_x = 100
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
t0 = 0.
if is_diff is True : 
    dt = 0.0005
    tmax = .25 
else :
    dt = 0.001
    tmax = 1.
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)

# Initial Conditions
s_dev2, mu_offset = .3, -1.
u_background = 0.
u0_init  = np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
u0_init += u_background

# Data
if objective == 'CI' or objective == 'CI and MODEL':
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

def limiters(u,i,m,treshold = None):
    '''
    force the solution to remain in a feasible domain
    '''
    if u[i] > m:
        u[i] = m
    if treshold is None:
        if u[i] < -m:
            u[i] = -m
    else:
        if u[i]<treshold:
            u[i] = treshold


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

### Adjoint
#  Construction of individual operators:

operator_advection_data         = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c_data)
operator_NL_advection_data      = lambda u,x,t,dt: st.operator_NL_advection(u,x,t,dt,p=1.)
operator_diffusion_data         = lambda u,x,t,dt: st.operator_diffusion(u,x,t,dt,p=nu_data)        # 
operator_dxxx_data              = lambda u,x,t,dt: st.operator_dxxx(u,x,t,dt,p=nu_data)        # 
#
    #  Construction of the operators:
operators_data = []
if is_adv is True :
    operators_data += [operator_advection_data]
if is_nl_adv is True :
    operators_data += [operator_NL_advection_data]
if is_diff is True :
    operators_data += [operator_diffusion_data]
if is_dxxx is True:
    operators_data += [operator_dxxx_data]




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



######################## DATA
U_data,t_sampled_data = [],[]
n_save = 1
u = u_data.copy()
# Loop
for i_t in xrange(n_it):
    u = st.integration(u,x,time[i_t],dt,operators_data,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False)
    #
    if verbose_data is True:
        if i_t%int(n_it/10.) == 0:
            s = 'computations data: ' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
            print(s)
        #
    #
    if i_t%n_save == 0:#save only one solution every n_save 
        U_data.append(u.copy())
        t_sampled_data.append(time[i_t])



def physics(c_0,nu,u0):
    '''
    # Compute the solution of the model.
    '''
    ##################################################
    # define the equation for the model:
    ##################################################
    #  Construction and update of individual operators:
    operator_advection              = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c_0)
    operator_NL_advection           = lambda u,x,t,dt: st.operator_NL_advection(u,x,t,dt,p=1.)
    operator_diffusion              = lambda u,x,t,dt: st.operator_diffusion(u,x,t,dt,p=nu)
    operator_dxxx                   = lambda u,x,t,dt: st.operator_dxxx(u,x,t,dt,p=1.)
    #
    ########
    #  Construction of the operators:
    operators = []
    if is_adv is True :
        operators += [operator_advection]
    if is_nl_adv is True :
        operators += [operator_NL_advection]
    if is_diff is True :
        operators += [operator_diffusion]
    if is_dxxx is True :
        operators += [operator_dxxx]
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
    ##################################################
    # define the equation for the model:
    ##################################################
    #  Construction and update of individual operators:
    operator_advection_adjoint      = lambda lambda_,u,x,t,dt: st.operator_advection_adjoint(lambda_,u,x,t,dt,p=c_0)
    operator_NL_advection_adjoint   = lambda lambda_,u,x,t,dt: st.operator_NL_advection_adjoint(lambda_,u,x,t,dt,p=1.)
    operator_diffusion_adjoint      = lambda lambda_,u,x,t,dt: st.operator_diffusion_adjoint(lambda_,u,x,t,dt,p=nu)  
    operator_dxxx_adjoint           = lambda lambda_,u,x,t,dt: st.operator_dxxx_adjoint(lambda_,u,x,t,dt,p=1.)
    #
    ########
    #  Construction of the operators:
    operators_adjoint = []
    if is_adv is True :
        operators_adjoint += [operator_advection_adjoint]
    if is_nl_adv is True :
        operators_adjoint += [operator_NL_advection_adjoint]
    if is_diff is True :
        operators_adjoint += [operator_diffusion_adjoint]
    if is_dxxx is True :
        operators_adjoint += [operator_dxxx_adjoint]
    #
    lambda0 = np.zeros(n_x)
    # lambda
    if 1:
        lambdas = []
        n_save = 1
        lambda_ = lambda0.copy()
        # Loop
        for i_t in xrange(n_it-1,-1,-1):
            lambda_ = st.integration_backward_mat(lambda_,U[i_t],x,time[i_t],-dt,U_data[i_t],operators=operators_adjoint,rhs_forcing = rhs_adjoint,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False)
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
        U,_ = physics(c_0,nu,lambdas)

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

U_data,_ = physics(c_data,nu_data,u_data)


for i_adjoint in range(n_adj_max):
    if i_adjoint >0 :
        learning_factor_ci = 1./(1.+(1.*i_adjoint)**.1)
        learning_factor_model = 50./(1.+(1.*i_adjoint)**.1)
    else :
        learning_factor_ci = .5
        learning_factor_model = 1.
    #
    if i_adjoint>1:
        if np.linalg.norm(DJ)<eps_DJ:
            print('gradient is small: break')
            break
    if i_adjoint>2:
        if np.linalg.norm(gradients[-1] - gradients[-2]) < eps_delta_DJ:
            print('gradient update is small: break')
            break
    ## update model:
    # update all parameters with the gradient
    if objective == 'MODEL' :
        q = q + 1.*DJ*learning_factor_model
    if objective == 'CI' :
        q = q + 1.*DJ*learning_factor_ci
    if objective == 'CI and MODEL' :
        q[0:n_x] = q[0:n_x] + 1.*DJ[0:n_x]*learning_factor_ci
        q[-2:] = q[-2:] + 1.*DJ[-2:]*learning_factor_model
    #
    # update model parameters
    if objective == 'CI':
        c_0,nu = q_physics[0],q_physics[1]
    else:
        limiters(q,-2,10.)
        limiters(q,-1,1.,.1)
        c_0,nu = q[-2],q[-1]
    #
    # update initial conditions:
    if objective == 'CI' or objective == 'CI and MODEL':
        for i_x in range(n_x): limiters(q,i_x,10.)
        u0 = q[:n_x]
    else:
        u0 = u0_init
    #
    ##################################################
    # Run the model
    ##################################################
    U,_ = physics(c_0,nu,u0)
    if i_adjoint == 0 :
        U_init = [u.copy() for u in U]
    ##################################################
    # Compute the gradient
    ##################################################
    # adjoint states
    lambdas = adjoint(c_0,nu,u0,U_data,U)
    # gradient
    DJ = gradient(c_0,nu,u0,U_data,U,lambdas)
    ########################################################
    ################### PRINTS #############################
    if verbose_minimization is True:
        s = '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print(s)
        s  = 'iteration : ' + str(i_adjoint)
        print(s)
        #
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







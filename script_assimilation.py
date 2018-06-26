import solver_matrix_tools as st
import numpy as np
import plot_tools as pt
import matplotlib.pyplot as plt
import matplotlib.animation as animation


##################################################
##################################################
# parameters
##################################################
##################################################
# Data Assimilation objective:
objectives = ['CI and MODEL','MODEL','CI']
objective = objectives[0]

# integration
method = 'RK4'

# physics
if objective == 'CI':
    q_physics= [6.5 , 0.45]
else:
    q_physics= [7.2 , 0.35]
#
q_data = [6.5,  0.45]
c_0,nu = q_physics[0],q_physics[1]
c_data,nu_data = q_data[0],q_data[1]

#space
n_x = 100
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
t0,dt = 0., 0.00025
tmax = .25
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)

# Initial Conditions
s_dev2, mu_offset = .3, -1.
u_background = 0.
u0_init  = np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
u0_init += u_background


# Data
if objective == 'CI' or objective == 'CI and MODEL':
    s_dev2_data, mu_offset_data = .2, 1.
    u_background_data = 0.
    u_data  = np.exp(- (x-mu_offset_data)**2 / (2. * s_dev2_data) ) / np.sqrt(2. * np.pi * s_dev2_data)
    u_data += u_background_data
    #u_data = np.sin(2.*np.pi*x) #for an harder problem
else: # if CI are similar
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
verbose = False



##################################################
##################################################
# define the equation:
##################################################
##################################################


def rhs(u,x,t,dt):
    return np.zeros(u.size)

def rhs_adjoint(lambda_,u,x,t,dt,p):
    '''
    this adjoint operator O is the RHS part of the adjoint equation, the linearization of the cost w.r.t. u:
    O(lambda,u,x,t,p) = dj/du
    '''
    #return np.zeros(lambda_.size)
    return 2. * (u-p) # j = ||u-y|| ^2




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

for i_adjoint in range(25):
    if i_adjoint>1:
        if np.linalg.norm(DJ)<1.e-6:
            print('gradient is small: break')
            break
    if i_adjoint>2:
        if np.linalg.norm(gradients[-1] - gradients[-2]) < 1.e-5:
            print('gradient update is small: break')
            break
    # update model:
    q= q + 1.*DJ
    if objective == 'CI':
        c_0,nu = q_physics[0],q_physics[1]
    else:
        c_0,nu = q[-2],q[-1]
    # update initial conditions:
    if objective == 'CI' or objective == 'CI and MODEL':
        u0 = q[:n_x]
    else:
        u0 = u0_init
    #
    lambda0 = np.zeros(n_x)
    #
    #######
    #  Construction of individual operators:
    operator_advection              = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c_0)
    operator_NL_advection           = lambda u,x,t,dt: st.operator_NL_advection(u,x,t,dt,p=1.)
    operator_diffusion              = lambda u,x,t,dt: st.operator_diffusion(u,x,t,dt,p=nu)    
    operator_advection_data         = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c_data)
    operator_NL_advection_data      = lambda u,x,t,dt: st.operator_NL_advection(u,x,t,dt,p=1.)
    operator_diffusion_data         = lambda u,x,t,dt: st.operator_diffusion(u,x,t,dt,p=nu_data)    
    operator_advection_adjoint      = lambda lambda_,u,x,t,dt: st.operator_advection_adjoint(lambda_,u,x,t,dt,p=c_0)
    operator_NL_advection_adjoint   = lambda lambda_,u,x,t,dt: st.operator_NL_advection_adjoint(lambda_,u,x,t,dt,p=1.)
    operator_diffusion_adjoint      = lambda lambda_,u,x,t,dt: st.operator_diffusion_adjoint(lambda_,u,x,t,dt,p=nu)  
     
    #
    ########
    #  Construction of the operators:
    operators = [
                    operator_advection,
                    #operator_NL_advection,
                    operator_diffusion,
                ]
    #
    # 
    operators_data = [
                    operator_advection_data,
                    #operator_NL_advection_data,
                    operator_diffusion_data,
                ]
    #
    #
    operators_adjoint = [
                    operator_advection_adjoint,
                    #operator_NL_advection_adjoint,
                    operator_diffusion_adjoint,
                ]
    #
    #
    ##################################################
    ##################################################
    # Compute the solution
    ##################################################
    ##################################################
    # Initialize solution
    U,t_sampled = [],[]
    n_save = 1
    u = u0.copy()
    # Loop
    for i_t in xrange(n_it):
        u = st.integration(u,x,time[i_t],dt,operators,method=method,bc_type = bc_type, bcs = bcs,return_rhs = False)
        if verbose is True:
            if i_t%int(n_it/10.) == 0:
                s = 'computations U:' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
                print(s)
            #
        #
        if i_t%n_save == 0:#save only one solution every n_save 
            U.append(u.copy())
            t_sampled.append(time[i_t])
        #
    #
    #
    if i_adjoint<1: # data computed only one time
        U_data,t_sampled_data = [],[]
        n_save = 1
        u = u_data.copy()
        # Loop
        for i_t in xrange(n_it):
            u = st.integration(u,x,time[i_t],dt,operators_data,method=method,bc_type = bc_type, bcs = bcs,return_rhs = False)
            if verbose is True:
                if i_t%int(n_it/10.) == 0:
                    s = 'computations data: ' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
                    print(s)
                #
            #
            if i_t%n_save == 0:#save only one solution every n_save 
                U_data.append(u.copy())
                t_sampled_data.append(time[i_t])
            #
        #
    #
    ##################################################
    ##################################################
    # Compute the gradient
    ##################################################
    ##################################################
    #
    # lambda
    if 1:
        lambdas = []
        n_save = 1
        lambda_ = lambda0.copy()
        # Loop
        for i_t in xrange(n_it-1,-1,-1):
            lambda_ = st.integration_backward_mat(lambda_,U[i_t],x,time[i_t],-dt,U_data[i_t],operators=operators_adjoint,rhs_forcing = rhs_adjoint,method=method,bc_type = bc_type, bcs = bcs,return_rhs = False)
            if verbose is True:
                if i_t%int(n_it/10.) == 0:
                    s = 'computations adjoint:' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
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
        dqj = [null_dqj for i in range(n_q)]
        dqf = [dqf_c,dqf_nu]
        dqg = [null_dqg for i in range(n_q)]
    #
    if objective == 'CI':
        dqj = [null_dqj for i in range(n_q)]
        dqf = [null_dqf for i in range(n_q)] 
        dqg = [( lambda i: lambda u,q: ci_dqg(u,q,i) )(i) for i in range(n_q)]
    #
    if objective == 'CI and MODEL':
        dqj = [null_dqj for i in range(n_q)]
        dqf = [null_dqf for i in range(n_x)] + [dqf_c,dqf_nu]
        dqg = [( lambda i: lambda u,q: ci_dqg(u,q,i) )(i) for i in range(n_q)]
    #
    DJ = st.gradient_q(u0,U,lambdas,U_data,x,time,q,dqj,dqf,dqg,mu)
    #
    ########################################################
    ################### PRINTS #############################
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
        s += '%.2f' % q[-2]
        s += ' nu: '
        s += '%.2f' % q[-1]
        print(s)
        s  = 'Gradient DJ (model) is:'
        s += '( %.4f ,' % DJ[-2]
        s += ' %.4f )' % DJ[-1]
        print(s)
    if  (objective=='MODEL') is False:
        s  = 'ecart CI: '
        s += '%.4f' % (np.linalg.norm(u0 + DJ[:n_x] - u_data))
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
#
##################################################
##################################################
# Plot the solution
##################################################
##################################################
#
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


##################################################
##################################################
# Plot the adjoint
##################################################
##################################################
#
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

##################################################
##################################################
# Plot the adjoint + burger
##################################################
##################################################
#
#
if 0:
    def update_line(num, lambdas, U,V,x,line1,line2,line3,verbose = True):
        '''
        update the plot
        '''
        line1.set_data(x,lambdas[num]+.01)
        line2.set_data(x,U[num])
        line3.set_data(x,V[num])
        if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
        return line1,line2,line3
    #
    #
    fig3, ax3 = plt.subplots()
    #
    line21, = ax3.plot([], [], 'r-')
    line22, = ax3.plot([], [], 'k-')
    line23, = ax3.plot([], [], 'g-')
    plt.xlim(x.min(),x.max())
    plt.ylim(lambdas[0].min()-1, lambdas[0].max()+1)
    plt.title('adjoint, burgers')
    plt.xlabel('x')
    plt.ylabel('$\lambda$,$u$')
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig3, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(lambdas,U,U_data,x,line21,line22,line23),
                                       interval=100, blit=True)
    #
    #
    plt.show()



if 0:
    pt.closeall()
    pt.multiplot2(
        (x,x,x,x),
        (U[0],U_data[0],lambdas[-1], (-U[0] + U_data[0])*2.),
        pt.Paradraw(colors = ['k','r','k','g'],
            marks = ['-','-','--','--'],
            thickness = [3,3,3,2],
            legend = ['u','y','$\lambda$','y-u'],
            )
        )



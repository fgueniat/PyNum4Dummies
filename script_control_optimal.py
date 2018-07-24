import solver_matrix_tools as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plot_tools as pt

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
n_adj_max = 15
eps_DJ = 1.e-5
eps_delta_DJ = 1.e-4
islearning = True

# type of integration
integration_method = 'RK4'

# physics
c_0 = 5.5
rho = .0

#space
n_x = 500
xmin,xmax = -12.,8.
x = np.linspace(xmin,xmax,n_x)

# time
t0,dt = 0., 0.003
tmax = 2.1
tmax = np.minimum(tmax,(xmax-xmin)/c_0)
n_it = int(tmax/dt)
n_q = n_it
time = t0 + dt*np.arange(n_it)

# Initial Conditions
s_dev2, mu_offset = .3, -10.
u_background = 1.1
u0_init  = np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
s_dev2, mu_offset = .5, -7.
u0_init  += np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
u0_init += u_background

u_obj = 1.0 * np.ones(x.shape)
# initial forcing
q = np.zeros(n_q)


# Boundary conditions
bc_type = 'dirichlet'
bcs = [1.1,1.1]
if bc_type == 'periodic': tmax = np.minimum(tmax,(xmax-xmin)/c_0)
u_objective,_ = st.ghost(u_obj,x,bc_type = bc_type,bcs = bcs)
DJ = np.zeros(q.shape)

# saves
q_s = []
gradients = []
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
def rhs(u,x,t,dt):
    return np.zeros(u.size)

def rhs_adjoint(lambda_,u,x,t,dt,p):
    '''
    this adjoint operator O is the RHS part of the adjoint equation, the linearization of the cost w.r.t. u:
    O(lambda,u,x,t,p) = dj/du
    '''
    #return np.zeros(lambda_.size)
    rhs = np.zeros(lambda_.size)
    if t >= tmax +  2.*dt: #dt is negative 
        rhs += 2. * (u-u_objective)# ||u(T) - u_obj|| ^2
    #rhs +=  2. * (u-u_objective)# ||u(T) - u_obj|| ^2
    #rhs += rho * np.abs(p) # ||u-y|| ^2
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
    derivative of f w.r.t. control parameters
    '''
    if t[0] == i:
        return np.exp(- (x-mu_offset_control)**2 / (  2. * s_dev2_control) )
    else:
        return np.zeros(x.shape)


operator_advection              = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c_0)
operator_advection_adjoint      = lambda lambda_,u,x,t,dt: st.operator_advection_adjoint(lambda_,u,x,t,dt,p=c_0)
#    
operators_without_control = [
                    operator_advection,
                ]

operators_adjoint = [
                    operator_advection_adjoint,
                    #operator_NL_advection_adjoint,
                ]










############################################
##################################################
# Start the optimization:
##################################################
##################################################




for i_adjoint in range(n_adj_max):
    # print
    if verbose_minimization is True:
        print('iteration ' + str(i_adjoint) + ' over ' + str(n_adj_max) )
    # learning: to accelerate (if the prefix is >15) and stabilize the minimization
    if islearning is True : 
        learning_factor = 20./(1.+(1.*i_adjoint)**.3)
        if verbose_minimization is True:
            print('learning factor = ' + "%.2f" %learning_factor )    
    else: 
        learning_factor = 1.
    #########################################################
    ## Criterion to stop
    #
    if i_adjoint>1:
        if np.linalg.norm(DJ)<eps_DJ:
            print('gradient is small: break')
            break
    #
    if i_adjoint>2:
        if np.linalg.norm(gradients[-1] - gradients[-2]) < eps_delta_DJ:
            print('gradient update is small: break')
            break
    #
    ## update forcing:
    # update all parameters with the gradient
    q= q + 1.*DJ*learning_factor
    #
    ##################################################
    # Compute the solution
    ##################################################
    # Initialize solution
    U,t_sampled = [],[]
    n_save = 1
    u = u0_init.copy()
    # Loop
    for i_t in xrange(n_it):
        operator_forcing = lambda u,x,t,dt: operator_forcing_full(u,x,t,dt,q[i_t])
        operators = operators_without_control #+ [operator_forcing,]
        u = st.integration(u,x,time[i_t],dt,operators,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False,rhs_forcing = operator_forcing)
        #
        if verbose_computations is True:
            if i_t%int(n_it * show_results_percentage/100.) == 0:
                s = 'computations U:' + '%.0f' % (int(100.*i_t/n_it)) + '%'
                print(s)
            #
        #
        if i_t%n_save == 0:#save only one solution every n_save 
            U.append(u.copy())
            t_sampled.append(time[i_t])
        #
    if i_adjoint == n_adj_max-1: # out of the loop to compare solution and gradient 
        break
    #
    #
    ##################################################
    # Compute the gradient
    ##################################################
    # initialize lambda
    lambda0 = np.zeros(n_x)
    # lambda
    lambdas = []
    n_save = 1
    lambda_ = lambda0.copy()
    # Loop
    for i_t in xrange(n_it-1,-1,-1):
        lambda_ = st.integration_backward_mat(lambda_,U[i_t],x,time[i_t],-dt,p_forcing = None, operators=operators_adjoint,rhs_forcing = rhs_adjoint,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False)
        ###############
        ###############
        ##### Hack: flux limiter has to be implemented: borders of adjoints can be ugly
        if 1:
            n_hack = 10
            lambda_[:n_hack] = lambda_[n_hack]
            lambda_[-n_hack:] = lambda_[-n_hack]
        ##############
        ##############
        if verbose_computations is True:
            if i_t%int(n_it * show_results_percentage /100.) == 0:
                s = 'computations adjoint:' + '%.0f' % (int(100.*i_t/n_it)) + '%'
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
     # dqg
    null_dqg = lambda u,q: np.zeros(u.shape) 
    #
    def dqji(q,i):
        dqj = 2.*rho * q[i]
        return dqj
    #        
    operator_dqj = [( lambda i: lambda u,x,data,t,q: dqji(q,i) )(i) for i in range(n_q)]
    operator_dqf = [(lambda i: lambda u,x,t,q: dqf(u,x,t,q,i) )(i) for i in range(n_q)] 
    operator_dqg = [null_dqg for i in range(n_q)]
    #
    DJ = st.gradient_q(u0_init,U,lambdas,None,x,time,q,operator_dqj,operator_dqf,operator_dqg,mu)/np.abs(dt)
    #
    print('|U(T) - U_obj| = ' + "%.2f" %(np.linalg.norm(U[-1]-u_obj)) )
    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    #############################################################
    ############################################################
    # save
    #
    q_s.append(q.copy())
    #lambdas_s.append(lambdas[0].copy())
    gradients.append(DJ)
    #
#

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('Optimization is (hopefully) a sucess')



##################################################################
#################################################################
# Plots




# Plot objective and u(T):
if 0:
    pt.multiplot2((x,x),(U[-1],u_obj),pt.Paradraw(x_label = 'x',y_label = 'y',legend=['$u(T)$','$u_{obj}$'], colors = ['k','r']))

# Plot the adjoint + solution + forcage
if 1:
    def update_line(num, lambdas, U,x,line1,line2,line3,line4,verbose = True):
        '''
        update the plot
        '''
        u_obj_transp = u_obj.copy()
        xs1 = c_0 * (tmax-time[num] )
        xs = ((xs1)%10.) - 5.
        print xs,xs1
        ind = np.argmin(np.abs(x - xs))
        #u_obj_transp = np.r_[u_obj[ind:],u_obj[:ind]]
        u_obj_transp = u_obj.copy()
        line1.set_data(x,5.*lambdas[num]+.01)
        line2.set_data(x,U[num])
        line3.set_data(x,operator_forcing_full(None,x,None,dt,q[num]))
        line4.set_data(x,u_obj_transp)
        if verbose is True: 
            s = 'time at t_' + str(num) + ' : %.2f'%(t_sampled[num])
            print(s)
        #
        return line1,line2,line3,line4,
    #
    #
    fig3, ax3 = plt.subplots()
    #
    line1, = ax3.plot([], [], 'r-')
    line2, = ax3.plot([], [], 'g-')
    line3, = ax3.plot([], [], 'b-')
    line4, = ax3.plot([], [], 'k-')
    plt.xlim(x.min(),x.max())
    umax = np.max([
        np.max(np.abs(np.array(U))), 
        np.max( np.abs(np.array(lambdas))), 
        np.max(np.abs(q))
        ])
    plt.ylim(-umax,umax)
    plt.title('adjoint, burgers')
    plt.xlabel('x')
    plt.ylabel('$\lambda$,$u$')
    plt.legend(['5*adjoint','u','forcing','obj'])
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig3, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(lambdas,U,x,line1,line2,line3,line4,),
                                       interval=100, blit=True)
    #
    #
    plt.show()



#
# Plot the solution
if 0:
    def update_line(num, U,x,line1,verbose = True):
        '''
        update the plot
        '''
        line1.set_data(x,U[num])
        if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
        return line1,
    #
    #
    fig3, ax3 = plt.subplots()
    #
    line21, = ax3.plot([], [], 'r-')
    line22, = ax3.plot(x, u_obj, 'k-')
    plt.xlim(x.min(),x.max())
    plt.ylim(np.array(U).min()-1, np.array(U).max()+1)
    plt.title('burgers')
    plt.xlabel('x')
    plt.ylabel('$u$')
    plt.legend(['u'])
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig3, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(U,x,line21),
                                       interval=100, blit=True)
    #
    #
    plt.show()



'''
#recompute by hand the gradient

dj = np.zeros(n_q)
for i in range(n_q):
    dj[i] = np.dot(lambdas[i],dqf(U[i],x,[i,i],q,i)) + 2. * rho* (q-DJ)[i]

q2 = q - DJ+dj

if 1:
    ##################################################
    # Compute the solution
    ##################################################
    # Initialize solution
    U2 = []
    n_save = 1
    u = u0_init.copy()
    # Loop
    for i_t in xrange(n_it):
        operator_forcing = lambda u,x,t,dt: operator_forcing_full(u,x,t,dt,q2[i_t])
        operators = operators_without_control #+ [operator_forcing,]
        u = st.integration(u,x,time[i_t],dt,operators,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False,rhs_forcing = operator_forcing)
        #
        if verbose_data is True:
            if i_t%int(n_it/4.) == 0:
                s = 'computations U:' + '%.0f' % (4.*i_t/int(n_it/25)) + '%'
                print(s)
            #
        #
        if i_t%n_save == 0:#save only one solution every n_save 
            U2.append(u.copy())
        #

pt.closeall()
pt.multiplot1((U[-1],U2[-1],u_obj),pt.Paradraw(colors=['r','k','b'],legend=['u','u2','obj']))
pt.multiplot1((50*q2,q),pt.Paradraw(colors=['k','r'], legend=['q2','q']))



'''





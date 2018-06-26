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
# integration
method = 'RK4'

# physics
q        = [4.3 , 0.28]
q_target = [6.5, 0.45]
c_0,nu = q[0],q[1]
c_target,nu_target = q_target[0],q_target[1]

##################################################
##################################################
# define the advection equation:
##################################################
##################################################

# inviscid: du/dt = - c_0 du/dx




def operator_advection(u,x,t,dt,p=c_target):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = c du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD_mat(-p,u,x)

def operator_advection_adjoint(lambda_,u,x,t,dt,p=c_target):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = -c dlambda/dx
    It uses the LUD function from solver_tools
    minus sign comes from the adjoint (cancelled by -dlambda/dt)
    '''
    return  -st.LUD_mat(p,lambda_,x)


def operator_NL_advection(u,x,t,dt,p=1.):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = -u du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD_mat(-p*u,u,x)

def operator_NL_advection_adjoint(lambda_,u,x,t,dt,p=1.):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = u dlambda /dx
    It uses the LUD function from solver_tools
    minus sign comes from the adjoint (cancelled by -dlambda/dt)
    '''
    return  st.LUD_mat(-p*u,lambda_,x)



def operator_diffusion(u,x,t,dt,p=nu_target):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = d^2u/dx^2
    It uses the diffusion_mat function from solver_tools
    '''
    return  p*st.diffusion_mat(u,x)

def operator_diffusion_adjoint(lambda_,u,x,t,dt,p=nu_target):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = d^2lambda/dx^2
    It uses the diffusion_mat function from solver_tools
    minus sign as (-dx)^2 = dx^2, but -dlambda/dt
    '''
    return  p*st.diffusion_mat(lambda_,x)

def rhs(u,x,t,dt):
    return np.zeros(u.size)

def rhs_adjoint(lambda_,u,x,t,dt,p):
    '''
    this adjoint operator O is the RHS part of the adjoint equation, the linearization of the cost w.r.t. u:
    O(lambda,u,x,t,p) = dj/du
    '''
    #return np.zeros(lambda_.size)
    return 2. * (u-p)

DJ = np.zeros(2)
dstore = []

for i_adjoint in range(20):
    dstore.append(DJ)
    q = q + 5.*DJ
    c_0,nu = q[0],q[1]

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
    s_dev2, mu = .2, -1.
    u_background = 0.
    u0  = np.exp(- (x-mu)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
    u0 += u_background


    lambda0 = np.zeros(n_x)

    # Target Conditions
    s_dev2_target, mu_target = .2, -1.
    u_background_target = 0.
    u_target  = np.exp(- (x-mu_target)**2 / (2. * s_dev2_target) ) / np.sqrt(2. * np.pi * s_dev2_target)
    u_target += u_background_target

    #u_target = u0

    # Boundary conditions
    bc_type = 'periodic'
    bcs = None

    # plot parameters:
    ind_plot_start,ind_plot_end,n_plot_skip = 0,n_it,int(n_it/100)
    verbose = False




    #  Construction of the operators:
    operator_advection_it = lambda u,x,t,dt: operator_advection(u,x,t,dt,p=q[0])
    operator_NL_advection_it = lambda u,x,t,dt: operator_NL_advection(u,x,t,dt,p=1.)
    operator_diffusion_it = lambda u,x,t,dt: operator_diffusion(u,x,t,dt,p=q[1])    
    #  Construction of the operators:
    operators = [
                    operator_advection_it,
                    #operator_NL_advection_it,
                    operator_diffusion_it,
                ]

    #  Construction of the operators:
    operator_advection_data = lambda u,x,t,dt: operator_advection(u,x,t,dt,p=c_target)
    operator_NL_advection_data = lambda u,x,t,dt: operator_NL_advection(u,x,t,dt,p=1.)
    operator_diffusion_data = lambda u,x,t,dt: operator_diffusion(u,x,t,dt,p=nu_target)
      
    operators_data = [
                    operator_advection_data,
                    #operator_NL_advection_data,
                    operator_diffusion_data,
                ]

    operators_adjoint = [
                    operator_advection_adjoint,
                    #operator_NL_advection_adjoint,
                    operator_diffusion_adjoint,
                ]




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
        if i_t%n_save == 0:#save only one solution every n_save 
            U.append(u.copy())
            t_sampled.append(time[i_t])


    if 1:
        U_target,t_sampled_target = [],[]
        n_save = 1
        u = u_target.copy()
        # Loop
        for i_t in xrange(n_it):
            u = st.integration(u,x,time[i_t],dt,operators_data,method=method,bc_type = bc_type, bcs = bcs,return_rhs = False)
            if verbose is True:
                if i_t%int(n_it/10.) == 0:
                    s = 'computations data: ' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
                    print(s)
            if i_t%n_save == 0:#save only one solution every n_save 
                U_target.append(u.copy())
                t_sampled_target.append(time[i_t])

    ##################################################
    ##################################################
    # Compute the gradient
    ##################################################
    ##################################################
    #lambda0 = U[-1].copy()

    # lambda
    if 1:
        lambdas = []
        n_save = 1
        lambda_ = lambda0.copy()
        # Loop
        for i_t in xrange(n_it-1,-1,-1):
            lambda_ = st.integration_backward_mat(lambda_,U[i_t],x,time[i_t],-dt,U_target[i_t],operators=operators_adjoint,rhs_forcing = rhs_adjoint,method=method,bc_type = bc_type, bcs = bcs,return_rhs = False)
            if verbose is True:
                if i_t%int(n_it/10.) == 0:
                    s = 'computations adjoint:' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
                    print(s)
            if i_t%n_save == 0:#save only one solution every n_save 
                lambdas.append(lambda_.copy())

    lambdas = lambdas[::-1]
    #mu
    mu = lambdas[0].copy()
    # final integration

    null_dqj = lambda u,d,x,t,q: 0.
    dqj = [null_dqj,null_dqj]

    def dqf_c (U,x,t,q):
        uu = U[t[0]].copy()
        uu,xx = st.ghost(uu,x)
        M = st.LUD_mat(-1.,uu,xx) # LHS
        uu = np.dot(M,uu)
        u,_ = st.deghost(uu,xx)
        return u

    def dqf_nu (U,x,t,q):
        uu = U[t[0]].copy()
        uu,xx = st.ghost(uu,x)
        M = st.diffusion_mat(uu,xx) #LHS
        uu = np.dot(M,uu)
        u,_ = st.deghost(uu,xx)
        return u

    dqf = [dqf_c,dqf_nu]

    null_dqg = lambda u,q: np.zeros(u.size)
    dqg = [null_dqg,null_dqg]

    DJ = st.gradient(u0,U,lambdas,U_target,x,time,q,dqj,dqf,dqg,mu)

    s = '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    print(s)
    s  = 'iteration : ' + str(i_adjoint)
    print(s)

    s  = 'targets are:        '
    s += ' c: '
    s += '%.2f' % q_target[0]
    s += ' nu: '
    s += '%.2f' % q_target[1]
    print(s)
    s  = 'initial guesses are:'
    s += ' c: '
    s += '%.2f' % q[0]
    s += ' nu: '
    s += '%.2f' % q[1]
    print(s)
    s  = 'Gradient DJ is:'
    s += '( %.4f ,' % DJ[0]
    s += ' %.4f )' % DJ[1]
    print(s)


dstore = np.array(dstore)
dstore = dstore[1:]
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
    line_ani = animation.FuncAnimation(fig3, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(lambdas,U,U_target,x,line21,line22,line23),
                                       interval=100, blit=True)
    #
    #
    plt.show()



if 0:
    pt.closeall()
    pt.multiplot2(
        (x,x,x,x),
        (U[0],U_target[0],lambdas[-1], (-U[0] + U_target[0])*2.),
        pt.Paradraw(colors = ['k','r','k','g'],
            marks = ['-','-','--','--'],
            thickness = [3,3,3,2],
            legend = ['u','y','$\lambda$','y-u'],
            )
        )


if 0:
    pd = pt.Paradraw(
            y_scale = 'log')
    pt.multiplot1((np.abs(dstore[:,0]),np.abs(dstore[:,1])),pd)

if 0:
    pd = pt.Paradraw()
    pt.multiplot1((dstore[:,0],dstore[:,1]),pd)



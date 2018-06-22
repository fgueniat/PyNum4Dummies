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
c_0 = 2.
nu = .5
#space
n_x = 200
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
t0,dt = 0., 0.0005
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
s_dev2_target, mu_target = .25, 1.
u_background_target = 0.
u_target  = np.exp(- (x-mu_target)**2 / (2. * s_dev2_target) ) / np.sqrt(2. * np.pi * s_dev2_target)
u_target += u_background_target

# Boundary conditions
bc_type = 'periodic'
bcs = None

# plot parameters:
ind_plot_start,ind_plot_end,n_plot_skip = 0,n_it,int(n_it/100)
verbose = True


##################################################
##################################################
# define the advection equation:
##################################################
##################################################

# inviscid: du/dt = - c_0 du/dx




def operator_advection(u,x,t,dt):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = c du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD_mat(-c_0,u,x)

def operator_advection_adjoint(lambda_,u,x,t,dt):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = -c dlambda/dx
    It uses the LUD function from solver_tools
    minus sign comes from the adjoint (cancelled by -dlambda/dt)
    '''
    return  -st.LUD_mat(c_0,lambda_,x)


def operator_NL_advection(u,x,t,dt):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = -u du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD_mat(-u,u,x)

def operator_NL_advection_adjoint(lambda_,u,x,t,dt):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = u dlambda /dx
    It uses the LUD function from solver_tools
    minus sign comes from the adjoint (cancelled by -dlambda/dt)
    '''
    return  st.LUD_mat(-u,lambda_,x)



def operator_diffusion(u,x,t,dt):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = d^2u/dx^2
    It uses the diffusion_mat function from solver_tools
    '''
    return  nu*st.diffusion_mat(u,x)

def operator_diffusion_adjoint(lambda_,u,x,t,dt):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = d^2lambda/dx^2
    It uses the diffusion_mat function from solver_tools
    minus sign as (-dx)^2 = dx^2, but -dlambda/dt
    '''
    return  nu*st.diffusion_mat(lambda_,x)

def rhs(u,x,t,dt):
    return np.zeros(u.size)

def rhs_adjoint(lambda_,u,x,t,dt,p):
    '''
    this adjoint operator O is the RHS part of the adjoint equation, the linearization of the cost w.r.t. u:
    O(lambda,u,x,t,p) = dj/du
    '''
    #return np.zeros(lambda_.size)
    return 2. * (u-p)


#  Construction of the operators:
operators = [
                operator_advection,
                #operator_NL_advection,
                #operator_diffusion,
            ]

operators_adjoint = [
                operator_advection_adjoint,
                #operator_NL_advection_adjoint,
                #operator_diffusion_adjoint,
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
        u = st.integration(u,x,time[i_t],dt,operators,method=method,bc_type = bc_type, bcs = bcs,return_rhs = False)
        if verbose is True:
            if i_t%int(n_it/10.) == 0:
                s = 'computations data: ' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
                print(s)
        if i_t%n_save == 0:#save only one solution every n_save 
            U_target.append(u.copy())
            t_sampled_target.append(time[i_t])


#lambda0 = U[-1].copy()
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
def update_line(num, lambdas, U,x,line1,line2,verbose = True):
    '''
    update the plot
    '''
    line1.set_data(x,lambdas[len(lambdas)-1-num]+.01)
    line2.set_data(x,U[num])
    if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
    return line1,line2

#
fig3, ax3 = plt.subplots()
#
line21, = ax3.plot([], [], 'r-')
line22, = ax3.plot([], [], 'k-')
plt.xlim(x.min(),x.max())
plt.ylim(lambdas[0].min()-1, lambdas[0].max()+1)
plt.title('adjoint + burger')
plt.xlabel('x')
plt.ylabel('$\lambda$')
#
plt.ion()
#
line_ani = animation.FuncAnimation(fig3, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(lambdas,U,x,line21,line22),
                                   interval=100, blit=True)

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



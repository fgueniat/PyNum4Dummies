import numpy as np
#import plot_tools as pt
import solver_tools as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation


##################################################
##################################################
# parameters
##################################################
##################################################
# physics
nu = 0.01
c_0 = 2.

# integration
method = 'RK4'

#space
n_x = 500
xmin,xmax = -1.,1.
x = np.linspace(xmin,xmax,n_x)

# time
t0,dt = 0., 0.0005
tmax = 1.5
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)

# Initial Conditions
u0 = np.zeros(x.shape)

# Boundary conditions
bc_type = 'dirichlet'
bcs = [0.,0.]
def BCS(time):
    return [np.sin(2.*np.pi * time),0.]

# plot parameters:
ind_plot_start,ind_plot_end,n_plot_skip = 0,n_it,int(n_it/100)
# print and debug
verbose = True
is_test = False
collect_rhs = True



##################################################
##################################################
# define the equation to solve:
##################################################
##################################################


def operator_advection(u,x,t,dt):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = -c_0 du/dx
    It uses the LUD function from solver_tools
    '''   
    return st.LUD(-c_0, u, x)

def operator_NL_advection(u,x,t,dt):
    '''
    this operator O is the RHS part of a non-linear advection term:
    O(u,x,t) = -u du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD(-1.*u,u,x)

def operator_diffusion(u,x,t,dt,nu=nu):
    '''
    this operator O is the RHS part of the diffusion term:
    O(u,x,t) = nu d^2u/dx^2
    It uses the central_scheme function from solver_tools
    '''
    return nu*st.central_scheme(u,x)

def operator_rhs(u,x,t,dt):
    '''
    this operator O is the RHS part, it corresponds to a forcing.
    In this exemple, it is somehow a diffusive term:
    O(u,x,t) = 0.1 * u**2
    '''
    return 0.1 * u**2


#  Construction of the operators:
operators = [
                operator_advection,
                #operator_NL_advection,
                #operator_diffusion,
                #operator_rhs
            ]

if verbose is True: print 'construction of the operator: finished'












##################################################
##################################################
# Compute the solution
##################################################
##################################################
if verbose is True: print 'computations of the solution: start'
# Initialize solution
U,t_sampled = [],[]
if collect_rhs is True: RHS = []
n_save = 1
u = u0.copy()
percentage_computations = -1
# Loop


for i_t in xrange(n_it):
    #
    if verbose is True: # print advancement of computations
        if i_t%int(n_it/20.)==0:
            percentage_computations +=1
            print 'computations in progress: ',5*percentage_computations,'%'
        #
    #
    bcs = BCS(time[i_t])
    if collect_rhs is True:
        u,rhs = st.integration(u,x,time[i_t],dt,operators,method=method,bc_type = bc_type, bcs = bcs,return_rhs = collect_rhs)
    else:
        u = st.integration(u,x,time[i_t],dt,operators,method=method,bc_type = bc_type, bcs = bcs,return_rhs = collect_rhs)
    if i_t%n_save == 0:#save only one solution every n_save 
        U.append(u.copy())
        if collect_rhs is True:
            RHS.append(rhs.copy())
        t_sampled.append(time[i_t])
    #
#

if verbose is True: print 'computations of the solution: finished'





##################################################
##################################################
# Plot the solution
##################################################
##################################################
#
def update_line(num, U,x,line,verbose = True):
    '''
    update the plot
    '''
    line.set_data(x,U[num])
    if verbose is True: print 'time at t_','%.2s' % t_sampled[num],' : ',t_sampled[num], ' max U ','%.4s' % U[num].max()
    return line,

fig, ax = plt.subplots()

line, = ax.plot([], [], 'r-')
plt.xlim(x.min(),x.max())
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('u')


plt.ion()

line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(U,x,line),
                                   interval=100, blit=True)

plt.show()







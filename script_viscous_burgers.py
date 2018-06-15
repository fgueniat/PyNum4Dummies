import numpy as np
import solver_tools as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation


##################################################
##################################################
# parameters
##################################################
##################################################
# physics
nu = 0.2

# integration
method = 'RK4'

#space
n_x = 200
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
n_it = 20000
t0,dt = 0., 0.001
time = t0 + dt*np.arange(n_it)

# Initial Conditions
s_dev2, mu = .02, 0.
u_background = 0.
u0  = np.exp(- (x-mu)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
u0 += u_background

# Boundary conditions
bc_type = 'periodic'
bcs = None

# plot parameters:
ind_plot_start,ind_plot_end,n_plot_skip = 0,n_it,100

# print
verbose = True

##################################################
##################################################
# define the burger equation:
##################################################
##################################################

# viscous: du/dt = - u du/dx + nu d^2u /dx^2

def operator_NL_advection(u,x,t,dt):
    '''
    this operator O is the RHS part of the advection term:
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


#  Construction of the operators:
operators = [
                operator_NL_advection,
                operator_diffusion,
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
    u = st.integration(u,x,time[i_t],dt,operators,method=method,bc_type = bc_type, bcs = bcs,return_rhs = False)
    if i_t%n_save == 0:#save only one solution every n_save 
        U.append(u.copy())
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
    if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
    return line,

fig, ax = plt.subplots()

line, = ax.plot([], [], 'r-')
plt.xlim(x.min(),x.max())
plt.ylim(u0.min()-1, u0.max()+1)
plt.xlabel('x')
plt.ylabel('u')


plt.ion()

line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(U,x,line),
                                   interval=100, blit=True)

plt.show()


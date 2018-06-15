import numpy as np
import solver_tools as st
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
c_0 = 1.
#space
n_x = 200
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
t0,dt = 0., 0.0005
tmax = 2.5
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)

# Initial Conditions
s_dev, mu = .02, 0.
u_background = 0.
u0  = np.exp(- (x-mu)**2 / (2. * s_dev) ) / np.sqrt(2. * np.pi * s_dev)
u0 += u_background

# Boundary conditions
bc_type = 'periodic'
bcs = None

# plot parameters:
ind_plot_start,ind_plot_end,n_plot_skip = 0,n_it,int(n_it/100)



##################################################
##################################################
# define the advection equation:
##################################################
##################################################

# inviscid: du/dt = - c_0 du/dx

def operator_NL_advection(u,x,t,dt):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = -u du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD(-c_0,u,x)

#  Construction of the operators:
operators = [
                operator_NL_advection,
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
    if i_t%n_save == 0:#save only one solution every n_save 
        U.append(u.copy())
        t_sampled.append(time[i_t])







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


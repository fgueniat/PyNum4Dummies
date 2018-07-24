import numpy as np
import solver_matrix_tools as st
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
n_x = 150
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
t0,dt = 0., 0.0005
tmax = 5.
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)

# Initial Conditions
s_dev2, mu = .2, 0.
u_background = 1.
u0  = np.exp(- (x-mu)**2 / (2. * s_dev2) ) #/ np.sqrt(2. * np.pi * s_dev2)
u0 += u_background
#u0 = np.sin(1.*np.pi*x) + 1.5
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

def v(u):
    return u
def vp(u):
    return np.ones(u.size)

def cp(u,x,t,dt):
    c_p = v(u) + u*vp(u)    
    return c_p


def operator_NL_advection(u,x,t,dt):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = -u du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD_mat(-cp(u,x,t,dt),u,x)

def operator_forcing(u,x,t,dt):
    '''
    '''
    u_d = 1.- u
    if np.max(u)>1.:
        amp = np.max(u) - 1.
        ind = np.argmax(u)
        # ind = x.size - (11-ind)
        mu = x[ind]
        s_dev2 = .5
        u_d += - amp * np.exp(- (x-mu)**2 / (2. * s_dev2) )
    A = np.diag(u_d)
    return A

#  Construction of the operators:
operators = [
                operator_NL_advection,
                operator_forcing
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


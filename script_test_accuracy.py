import numpy as np
import plot_tools as pt
import solver_tools as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation



##################################################
##################################################
# parameters
##################################################
##################################################
# physics
nu = 0.0
c_0 = 2.
u_l,u_r,x_r = 4., 2.,1
alpha = (u_l - u_r) / x_r
def IC(x,ul=u_l,ur=u_r,xr=x_r):
    '''
    implement initial conditions
    '''
    u=np.zeros(x.shape)
    for i_x,x_i in enumerate(x):
        if x_i<0:       u[i_x] = ul
        elif x_i>x_r:   u[i_x] = ur
        else:           u[i_x] = u_l + x_i * (ur - ul)/xr
    return u

# integration
method = 'RK4'

#space
n_x = 500
xmin,xmax = -2.,3.
x = np.linspace(xmin,xmax,n_x)
def shift(u,ul=u_l,ur=u_r,nx = n_x):
    '''
    shift u so the center of the ramp stays in the middle of the domain
    '''
    ind = np.argmin(np.abs(u-0.5 * (ur+ul)))
    offset = int(nx/2) - ind
    if offset>0:    return np.r_[np.ones(offset) * ul, u[:-offset]]
    else:           return np.r_[u[-offset:],np.ones(-offset) * ur]

# time
t0,dt = 0., 0.0005
tmax = 1./alpha # time at which the solution becomes discontinuous
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)

# Initial Conditions
u0 = IC(x)

# Boundary conditions
bc_type = 'dirichlet'
bcs = [u_l,u_r]

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
                #operator_advection,
                operator_NL_advection,
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
    if collect_rhs is True:
        u,rhs = st.integration(u,x,time[i_t],dt,operators,method=method,bc_type = bc_type, bcs = bcs,return_rhs = collect_rhs)
    else:
        u = st.integration(u,x,time[i_t],dt,operators,method=method,bc_type = bc_type, bcs = bcs,return_rhs = collect_rhs)
    if i_t%n_save == 0:#save only one solution every n_save
        #shift
        u=shift(u)
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
plt.ylim(u0.min()-1, u0.max()+1)
plt.xlabel('x')
plt.ylabel('u')


plt.ion()

line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(U,x,line),
                                   interval=100, blit=True)

plt.show()





##################################################
##################################################
# Verifications
##################################################
##################################################


def test_advection_i(i,c=c_0,verbose=False):
    '''
    function that test the accuracy for the advection function at time i:
    du/dt = -c0 du/dx
    it checks, for a time i, the ratio r(x) = du/dt / du/dx, supposedly equal to -c0.
    '''
    dudt = (U[i+1] - U[i-1]) / (2 * n_save * dt)
    dudt = dudt[1:-1]
    dudx = (U[i][2:] - U[i][:-2])/((x[2:] - x[:-2]))
    if verbose is True:
        print 'error in % w.r.t. c',100*np.mean(dudt/dudx + c)/np.abs(c),'% +-',100*np.std(dudt/dudx + c)/np.abs(c),'%'
    if 0. in dudx:
        dudx += 0.000001*np.random.random(dudx.shape)
    return np.nanmean(dudt/dudx + c)/np.abs(c)
    #return dudt,dudx


def test_diffusion_i(i,nu = nu,verbose=False):
    '''
    function that test the accuracy for the advection function at time i:
    du/dt = nu d^2u/dx^2
    it checks, for a time i, the ratio r(x) = du/dt / d^2u/dx^2, supposedly equal to nu.
    '''
    dudt = (U[i+1] - U[i-1]) / (2 * n_save * dt)
    dudt = dudt[1:-1]
    d2udx2 = ( U[i][2:] - 2.*U[i][1:-1] + U[i][:-2] )/( 2.* (0.5*(x[2:] - x[:-2]))**2.)
    if verbose is True:
        print 'error in % w.r.t. dudt',100*np.mean(dudt/d2udx2 - nu)/np.abs(nu),'% +-',100*np.std(dudt/d2udx2 - nu)/np.abs(nu),'%'
    if 0. in dudx:
        d2udx2 += 0.000001*np.random.random(d2udx2.shape)
    return np.nanmean(dudt/d2udx2 - nu)/np.abs(nu)
    #return dudt,d2udx2

if is_test is True:
    names = [op.__name__ for op in operators]
    # 
    if names == [operator_advection.__name__]:
        errors = np.zeros(len(U)-2)
        for i in xrange(1,len(U)-1):
            errors[i-1] = test_advection_i(i)
        print 'errors advection',np.nanmean(errors),np.nanstd(errors)





import solver_matrix_tools as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''

This script shows how to use the solver library, to solve partial differential equations.
the user defines the physics (see script_pde.py)

'''

##################################################
##################################################
# parameters
##################################################
##################################################

# type of integration
integration_method = 'RK4'

q_physics= [-6.5 , 0.35]

c_0,nu = q_physics[0],q_physics[1]

#space
n_x = 100
xmin,xmax = -5.,5.
x = np.linspace(xmin,xmax,n_x)

# time
t0,dt = 0., 0.001
tmax = 10.
n_it = int(tmax/dt)
time = t0 + dt*np.arange(n_it)


# np.sqrt(np.sum((U_data[-1] - u_exact(x,1))**2)/n_x)
# Boundary conditions
BC_TYPES = ['periodic','dirichlet']
BC_FORCING = [False,True]
bc_type = BC_TYPES[1]
bc_f = BC_FORCING[1]
if bc_type == 'periodic' :
    bcs = None
else :
    bcs = [0,0.]

# Initial Conditions
if bc_type == 'periodic' or bc_type == 'neumann' :
    s_dev2, mu_offset = .3, -1.
    u_background = 0.
    u0_init  = 2.*np.exp(- (x-mu_offset)**2 / (2. * s_dev2) ) / np.sqrt(2. * np.pi * s_dev2)
    u0_init += u_background
    u0_init += .4 * np.sin(2.*np.pi * x)
else :
    '''
    In case of a RHS forcing + dirichlet boundary conditions:
    special case of burger equation, where the exact solution can be computed.
    For detals, see Furbish et al. 2008
    '''
    n_x = 100
    xmin,xmax = -1.,1.
    x = np.linspace(xmin,xmax,n_x)
    N_f = 3
    alpha_1_f,alpha_2_f = 0.02,0.015
    l_1_f,l_2_f = 2.,5.
    k_1_f,k_2_f,k_3_f = 2.,5.,11.
    C,C_1_f,C_2_f,C_3_f = 8.1,5.,1.,2.
    c_0,nu  = 1.,0.001
    def u_exact(x,t,N = N_f,C=[C,C_1_f,C_2_f,C_3_f],K=[k_1_f,k_2_f,k_3_f],L=[l_1_f,l_2_f],a = [alpha_1_f,alpha_2_f]):
        '''
        In case of a RHS forcing + dirichlet boundary conditions:
        special case of burger equation, where the exact solution can be computed (here).
        For detals, see Furbish et al. 2008
        '''
        u = np.zeros(x.size)
        u += a[0] * np.sin(L[0] * np.pi * x) 
        u += a[1] * np.sin(L[1] * np.pi * x) 
        utop = np.zeros(x.size)
        ubot = np.zeros(x.size)
        for i in range(N_f):
            utop += C[i+1] * K[i] * np.exp(-nu * (K[i] * np.pi)**2 * t) * np.sin(K[i] * np.pi * x)
            ubot += C[i+1] * np.exp(-nu * (K[i] * np.pi)**2 * t) * np.cos(K[i] * np.pi * x)
        u += 2. * nu * np.pi * (utop / (C[0] + ubot))
        return u
    u0_init = u_exact(x,0.)
    



# plot parameters:
n_save = 1
ind_plot_start,ind_plot_end,n_plot_skip = 0,int(1.*n_it/n_save),int(n_it/(100*n_save))
verbose = True



##################################################
##################################################
# define the equation:
##################################################
##################################################

### Physics
# general forcing:
def rhs(u,x,t,dt,N = N_f,C=[C_1_f,C_2_f,C_3_f],K=[k_1_f,k_2_f,k_3_f],L=[l_1_f,l_2_f],a = [alpha_1_f,alpha_2_f],p=1.):
    '''
    In case of a RHS forcing + dirichlet boundary conditions:
    special case of burger equation, where the exact solution can be computed.
    For detals, see Furbish et al. 2008
    '''
    # 
    r= np.zeros(x.size)
    if bc_f is True:
        r += -np.pi *   (                                               \
                            a[0] * np.sin(L[0] * np.pi * x) +           \
                            a[1] * np.sin(L[1] * np.pi * x)             \
                        ) *                                             \
                        (                                               \
                            a[0] * K[0] * np.cos(L[0] * np.pi * x) +    \
                            a[1] * K[1] * np.cos(L[1] * np.pi * x)      \
                        )
        r += nu * np.pi * np.pi *   (                                                           \
                                                a[0] * K[0] * K[0] * np.sin(L[0] * np.pi * x) + \
                                                a[1] * K[1] * K[1] * np.sin(L[1] * np.pi * x)   \
                                    )
        r += - a[0] * L[0] * np.pi * np.cos(L[0] * np.pi * x) * u
        r += - a[1] * L[1] * np.pi * np.cos(L[1] * np.pi * x) * u
        #
    #
    return p*r

#  Construction of individual operators:

operator_advection_data     = lambda u,x,t,dt: st.operator_advection(u,x,t,dt,p=c_0)

operator_u_advection_data   = lambda u,x,t,dt: st.operator_NL_advection(u,x,t,dt,p=1.)
operator_diffusion_data     = lambda u,x,t,dt: st.operator_diffusion(u,x,t,dt,p=nu)        # 

def operator_NL_advection_data(u,x,t,dt,a=[alpha_1_f,alpha_2_f],L=[l_1_f,l_2_f]):
    '''
    In case of a RHS forcing + dirichlet boundary conditions:
    special case of burger equation, where the exact solution can be computed.
    For detals, see Furbish et al. 2008
    '''
    c = a[0] * np.sin(L[0] * np.pi * x) +  a[1] * np.sin(L[1] * np.pi * x) 
    return st.LUD_mat(c,u,x)
#
operators_data = [
                    #operator_advection_data,
                    operator_u_advection_data,
                    operator_NL_advection_data,
                    operator_diffusion_data,
                ]



##################################################
##################################################
# Computations:
##################################################
##################################################



######################## DATA
U_data,t_sampled = [],[]
u = u0_init.copy()
# Loop
for i_t in xrange(n_it):
    u = st.integration(u,x,time[i_t],dt,operators_data,method=integration_method,bc_type = bc_type, bcs = bcs,return_rhs = False, rhs_forcing = rhs)
    #
    if verbose is True:
        if i_t%int(n_it/10.) == 0:
            s = 'computations data: ' + '%.0f' % (10.*i_t/int(n_it/10)) + '%'
            print(s)
        #
    #
    if i_t%n_save == 0:#save only one solution every n_save 
        U_data.append(u.copy())
        t_sampled.append(time[i_t])



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
    plt.title('solution of pde')
    plt.xlim(x.min(),x.max())
    plt.ylim(u0_init.min()-.1, u0_init.max()+.1)
    plt.xlabel('x')
    plt.ylabel('u')
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(U_data,x,line),
                                       interval=100, blit=True)
    #
    plt.show()


# Plot the solution and the exact solution for forcing case with dirichlet
if 0:
    def update_line(num, U,x,line1,line2,line3,verbose = True):
        '''
        update the plot
        '''
        t = t_sampled[num]
        u = u_exact(x,t)
        r = rhs(u,x,t,1.,p=1.)
        line1.set_data(x,U[num])
        line2.set_data(x,u)
        line3.set_data(x,r)
        if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
        return line1,line2,line3
    #
    fig, ax = plt.subplots()
    #
    line1, = ax.plot([], [], 'k-')
    line2, = ax.plot([], [], 'r--')
    line3, = ax.plot([], [], 'g--')
    ax.legend(['u','exact','forcing'])
    plt.title('solution of pde')
    plt.xlim(x.min(),x.max())
    plt.ylim(u0_init.min()-.1, u0_init.max()+.1)
    plt.xlabel('x')
    plt.ylabel('u')
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end,n_plot_skip), fargs=(U_data,x,line1,line2,line3),
                                       interval=100, blit=True)
    #
    plt.show()

if 1:
    def update_line(num, U,x,line1,line2,line3,verbose = True):
        '''
        update the plot
        '''
        t = t_sampled[num]
        dt = t_sampled[num+1] - t_sampled[num]
        dx = x[2]-x[1]
        u1 = U[num]
        u2 = U[num+1]
        r_app = (u2-u1)/dt
        #r_app[1:-1] += -nu * (u1[2:] + u1[:-2] - 2.*u[1:-1]) / (0.5 * (dx**2.))
        r_app += np.dot(st.diffusion_mat(u,x),u)
        #r_app[1:] += u[1:] * (u1[1:] - u1[0:-1]) / (dx)
        r_app += np.dot(st.LUD_mat(u,u,x),u)
        a = alpha_1_f * np.sin(l_1_f * np.pi * x) + alpha_2_f * np.sin(l_2_f * np.pi * x)
        #r_app[1:] += ((a*u1)[1:] - (a*u1)[0:-1]) / (dx)
        r_app += np.dot(st.LUD_mat(a,u,x),u)
        r_u = rhs(u1,x,t,1.,p=1.)
        r_e = rhs(u_exact(x,t),x,t,1.,p=1.)
        line1.set_data(x,r_e)
        line2.set_data(x,r_u)
        line3.set_data(x,r_app)
        if verbose is True: print 'time at t_',num,' : ',t_sampled[num]
        return line1,line2,line3
    #
    fig, ax = plt.subplots()
    #
    line1, = ax.plot([], [], 'k-')
    line2, = ax.plot([], [], 'k--')
    line3, = ax.plot([], [], 'r--')
    ax.legend(['forcing exact', 'forcing data','forcing reconstructed'])
    plt.title('solution of pde')
    plt.xlim(x.min(),x.max())
    plt.ylim(u0_init.min()-.1, u0_init.max()+.1)
    plt.xlabel('x')
    plt.ylabel('u')
    #
    plt.ion()
    #
    line_ani = animation.FuncAnimation(fig, update_line, xrange(ind_plot_start,ind_plot_end-1,n_plot_skip), fargs=(U_data,x,line1,line2,line3),
                                       interval=100, blit=True)
    #
    plt.show()






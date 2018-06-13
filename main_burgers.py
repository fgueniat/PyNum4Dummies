import numpy as np
import plot_tools as pt
import solver_tools as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation


c_0 = 10.
nu = 0.01
N=2000
ipstart,ipfin,ipskip = 0,2000,20
method = 'RK4'

def operator_NL_advection(u,x,t,dt):
    return  st.upwind(-c_0*u,u,x)

def operator_advection(u,x,t,dt):
    return st.upwind(-c_0, u, x)

def operator_advection2(u,x,t,dt):
    return st.LUD(-c_0, u, x)

def operator_diffusion(u,x,t,dt):
    return nu*st.central_scheme(u,x)

def operator_rhs(u,x,t,dt):
    return 0.1 * u**2

operators = [
        #operator_advection,
        #operator_advection2,
        #operator_NL_advection,
        operator_diffusion,
        #operator_rhs
        ]

nx = 200
x = np.linspace(-5,5,nx)
dx = x[1]-x[0]

tmin = 0
dt = 0.001
time = tmin + dt*np.arange(N)

#
s_dev, mu = .02, 0.
u_background = 0.
u0  = np.exp(- (x-mu)**2 / (2. * s_dev) ) / np.sqrt(2. * np.pi * s_dev)
u0 += u_background
s_dev, mu = .1, -2.
u0 += -1.*np.exp(- (x-mu)**2 / (2. * s_dev) ) / np.sqrt(2. * np.pi * s_dev)

#u0 = np.sin(np.pi*x) +1



U = []
t_sampled = []
u = u0.copy()
RHS = []
skip=1


for i_t in xrange(N):
    u,rhs=st.integration(u,x,time[i_t],dt,operators,method=method)
    if i_t%skip == 0: 
        U.append(u.copy())
        RHS.append(rhs.copy())
        t_sampled.append(time[i_t])



def verif_advection(i=int(N/2),c=c_0):
    dudt = (U[i+1] - U[i-1]) / (2*skip * dt)
    dudt = dudt[1:-1]
    dudx = (U[i][2:] - U[i][:-2])/((x[2:] - x[:-2]))
    print 'error in % w.r.t. c',100*np.mean(dudt/dudx + c)/np.abs(c),'% +-',100*np.std(dudt/dudx + c)/np.abs(c),'%'
    return dudt,dudx


def verif_diffusion(i=int(N/2),nu = nu):
    dudt = (U[i+1] - U[i-1]) / (2*skip * dt)
    dudt = dudt[1:-1]
    d2udx2 = ( U[i][2:] - 2.*U[i][1:-1] + U[i][:-2] )/( 2.* (0.5*(x[2:] - x[:-2]))**2.)
    print 'error in % w.r.t. dudt',100*np.mean(dudt/d2udx2 - nu)/np.abs(nu),'% +-',100*np.std(dudt/d2udx2 - nu)/np.abs(nu),'%'
    return dudt,d2udx2



def update_line(num, U,x,line):
    line.set_data(x,U[num])
    print 'time at t_',num,' : ',t_sampled[num]
    return line,

fig, ax = plt.subplots()

line, = ax.plot([], [], 'r-')
plt.xlim(x.min(),x.max())
plt.ylim(u0.min()-1, u0.max()+1)
plt.xlabel('x')
plt.ylabel('u')


plt.ion()

line_ani = animation.FuncAnimation(fig, update_line, xrange(ipstart,ipfin,ipskip), fargs=(U,x,line),
                                   interval=100, blit=True)

plt.show()


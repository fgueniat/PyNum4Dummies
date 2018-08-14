import solver_matrix_tools as st
'''
library that contains a few useful, already discretized operators and their adjoint.

- advection $c \\partial_x u$
- diffusion $nu \\partial_{x,x} u$
- non linear advection: $u\\partial_x u $

'''

def operator_advection(u,x,t,dt,p=1):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = c du/dx
    It uses the LUD function from solver_tools
    '''
    return  st.LUD_mat(-p,u,x)

def operator_advection_adjoint(lambda_,u,x,t,dt,p=1):
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


def operator_diffusion(u,x,t,dt,p=1):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = d^2u/dx^2
    It uses the diffusion_mat function from solver_tools
    '''
    return  p*st.diffusion_mat(u,x)

def operator_diffusion_adjoint(lambda_,u,x,t,dt,p=1):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = d^2lambda/dx^2
    It uses the diffusion_mat function from solver_tools
    minus sign as (-dx)^2 = dx^2, but -dlambda/dt
    '''
    return  p*st.diffusion_mat(lambda_,x)

def operator_dxxx(u,x,t,dt,p=1):
    '''
    this operator O is the RHS part of the advection term:
    O(u,x,t) = d^3u/dx^3
    It uses the dxxx_mat function from solver_tools
    '''
    return  p*st.dxxx_mat(u,x)

def operator_dxxx_adjoint(lambda_,u,x,t,dt,p=1):
    '''
    this adjoint operator O is the RHS part of the advection term:
    O(lambda,u,x,t) = d^3lambda/dx^3
    It uses the dxxx_mat function from solver_tools
    '''
    return  -p*st.dxxx_mat(lambda_,x)



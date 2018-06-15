# PyNum4Dummies

This project consists in a library solver_tools and a bunch of python scripts.

The library aims at solving "easy" 1d partial differential equations (pdes), for education and reasearach.
Exemples of these pdes are:
* The heat equation
* Burger equation
* The advection equation

The librairie contains:
* a function to integrate the equation
* discretization of:
	** space derivative:
    	- LUD
     	- upwind
	** second space derivative
    	- central_scheme
* cfl (compute and print the cfl number)

Other functions will be implemented, to derive reduced order models (ROM) of the equation, as well as computing the adjoint variables (see later).

The various scripts illustrate the use of the library.

Todo:
* advection: instabilities in LUD when too stiff ? Flux limiter ?
* adjoint.






Computing the gradient of the cost functional

We want to look at the problem depending on some parameters:

$$
J(q) = \int_0^T 
		j(u,q,t)	
	{\textrm d}t 
$$
s.t. the physics:
$$
F(u,\dot{u},q,t) = 0
$$
and considering that the initial conditions are (potentially) related to the parameters q with:
$$
g(u(0),q) = 0
$$

We want to find the argmin of J.

Identifying the minimum of the $J$ relies on the gradient of the functional with respect to the parameters:
$D_q J = \frac{D\,J}{Dq}$. 

Finite differentiation is out of reach if the size of q is large.

For that, one can introduce the Lagrangian $\mathfrac{L}$, associated with the two Lagrange parameters $\boldsymbol{\lambda}$ and $\boldsymbol{\mu}$:

$$
\begin{split}
  &\mathfrac{L} 
			\bigg( u,\dot{u},\boldsymbol{q},\boldsymbol{\lambda},\boldsymbol{\mu} \bigg)
  = \\
  & \qquad \int_0^T
  	\bigg[
		j(u,\boldsymbol{q},t) + \boldsymbol{\lambda}^T F(u,\dot{u},\boldsymbol{q},t)
  	\bigg] {\textrm d}t \\
   & \qquad + \boldsymbol{\mu}^T \left(  g(u(0),\boldsymbol{q})  \right),
\end{split}
$$
where $^T$ is the transpose operator. 
Naturally, both $u$ and $\dot{u}$ are considered as variables.

As $F$ and $g$ are null by construction, $\boldsymbol{\lambda}$ and $\boldsymbol{\mu}$ can be designed specifically to alleviate the computations.

The gradient $D_q\mathfrac{L} = \frac{D\mathfrac{L}}{Dq}$ of $\mathfrac{L}$ is
$$
	\begin{split}
		&D_q\mathfrac{L}\mypar{u,\dot{u},\boldsymbol{q},\boldsymbol{\lambda},\boldsymbol{\mu}} = \\
		& \qquad \int_0^T  \bigg[\partial_{u}j(u,\boldsymbol{q},t)d_{\boldsymbol{q}} u  +\partial_{\boldsymbol{q}} j(u,\boldsymbol{q},t)  \\
		& \qquad + \boldsymbol{\lambda}^T \partial_{u}  F(u,\dot{u},\boldsymbol{q},t) {\textrm d}_{\boldsymbol{q}} u \\
		& \qquad + \boldsymbol{\lambda}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t)d_{\boldsymbol{q}} \dot{u}  \\
		& \qquad + \boldsymbol{\lambda}^T\partial_qF(u,\dot{u},\boldsymbol{q},t) \bigg] {\textrm d}t \\
		& \qquad  + \boldsymbol{\mu}^T \partial_{u(0)} g(u(0),\boldsymbol{q}) {\textrm d}_q{u(0)}  \\
		& \qquad + \boldsymbol{\mu}^T \partial_{\boldsymbol{q}} g(u(0),\boldsymbol{q}).
	\end{split}
$$
Upon optimality, one has $D_q\mathfrac{L} = D_{\boldsymbol{q}} J$.

The term in $d_{\dot{u}}$ cannot be easily estimated. An integration by parts gives:

\begin{equation*}
	\begin{split}
		&\int_0^T\boldsymbol{\boldsymbol{\lambda}}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t)d_{\boldsymbol{q}} \dot{u} {\textrm d}t = \\
		&\qquad \left[\boldsymbol{\lambda}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t) {\textrm d}_{\boldsymbol{q}} u \right]_0^T \\
		& \qquad - \int_0^T \left\{\dot{\boldsymbol{\lambda}}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t) \right. \\
		& \qquad \left. + \boldsymbol{\lambda}^T \partial_t \partial_{\dot{u}}F(u,\dot{u},\boldsymbol{q},t)\right\}d_{\boldsymbol{q}} u {\textrm d}t.
	\end{split}
\end{equation*}
The term associated with ${\textrm d}_{\dot{u}}$ can now be replaced.

Ordering terms leads to:

$$
	\begin{split}
		&D_q\mathfrac{L}\left(u,\dot{u},\boldsymbol{q},\boldsymbol{\lambda},\boldsymbol{\mu}\right) = \\
		& \qquad \int_0^T  \Bigg[  \bigg(\partial_{u}j\mypar{u,\boldsymbol{q},t}  +  \boldsymbol{\lambda}^T \partial_{u}  F\mypar{u,\dot{u},\boldsymbol{q},t} \\
		& \qquad - \big\{\dot{\boldsymbol{\lambda}}^T\partial_{\dot{u}} F\mypar{u,\dot{u},\boldsymbol{q},t} \\
		& \qquad +  \boldsymbol{\lambda}^T \partial_t\partial_{\dot{u}}F\mypar{u,\dot{u},\boldsymbol{q},t}\big\}\bigg) {\textrm d}_{\boldsymbol{q}} u   \\
		& \qquad + \partial_{\boldsymbol{q}} j(u,\boldsymbol{q},t)    + \boldsymbol{\lambda}^T\partial_qF(u,\dot{u},\boldsymbol{q},t) \Bigg] {\textrm d}t  \\
		& \qquad + \bigg(\boldsymbol{\mu}^T \partial_{u} g(u,\boldsymbol{q}) \\
		& \qquad -  \left. \boldsymbol{\lambda}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t) \bigg)
		\right|_0 dq_{u(0)}  \\
		& \qquad +  \boldsymbol{\mu}^T \partial_{\boldsymbol{q}} g(u(0),\boldsymbol{q}) \\
		& \qquad +  \left.\bigg(\boldsymbol{\lambda}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t)\bigg)\right|_T {\textrm d}_{\boldsymbol{q}} u(T)
	\end{split}
$$

Proper choices for $\boldsymbol{\lambda}$ and $\boldsymbol{\mu}$ allow to simplify the expression of $D_q\mathfrac{L}$.
The choice of $\boldsymbol{\lambda}\mypar{T} = \boldsymbol{0}$ nullifies the term $\left.\bigg(\boldsymbol{\lambda}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t)\bigg)\right|_T {\textrm d}_{\boldsymbol{q}} u(T)$.
$\boldsymbol{\lambda}$ can then be chosen as the solution of the so-called adjoint equation,~\cite{Ledimet1986,Talagrand1997}:
$$
	\begin{split}
		& \bigg(\partial_{u}j(u,\boldsymbol{q},t) + \boldsymbol{\lambda}^T \partial_{u}  F(u,\dot{u},\boldsymbol{q},t) \\
		& \qquad - \big\{\dot{\boldsymbol{\lambda}}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t) \\
		& \qquad + \boldsymbol{\lambda}^T \partial_t\partial_{\dot{u}}F(u,\dot{u},\boldsymbol{q},t)\big\}\bigg) = \boldsymbol{0},
	\end{split}
$$
integrated backwards in time. 
Finally, the Lagrange parameter $\boldsymbol{\mu}$ is set so that it nullifies the component associated with $dq_{u(0)}$:
$$ \left.\mya{\boldsymbol{\mu}^T \partial_{u} g(u,\boldsymbol{q}) - \boldsymbol{\lambda}^T\partial_{\dot{u}} F(u,\dot{u},\boldsymbol{q},t) }\right|_0 = \boldsymbol{0}.$$
Then, computing $D_q\mathfrac{L}$, hence $ D_qJ$, is achieved by the integration of:
$$
	\begin{split}
		& D_qJ\mypar{u,\boldsymbol{q}} = \\
		& \qquad \int_0^T  \bigg[  \partial_{\boldsymbol{q}} j(u,\boldsymbol{q},t)  + \boldsymbol{\lambda}^T\partial_qF(u,\dot{u},\boldsymbol{q},t) \bigg] {\textrm d}t \\
		& \qquad +   \boldsymbol{\mu}^T \partial_{\boldsymbol{q}} g(u(0),\boldsymbol{q}).
	\end{split}
$$



import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import pyomo.environ as pyo
from simulator.simulator_constants import dt

def LinearDynamicsModel(model,t):
    xPlus = np.array([model.x[0,t] + model.x[1,t]*dt, model.x[1,t]+model.u[0,t]*dt])
    return xPlus

def solve_cftoc(dynModel, R, P, N, x0, x_goal, nu, uL, uU):
    
    model = pyo.ConcreteModel()
    model.N = N
    model.nx = np.size(x0, 0)
    model.nu = nu
    
    # length of finite optimization problem:
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )  
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True  )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True  )
    
    # these are 2d arrays:
    model.R = R
    model.P = P
    
    # Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.tIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX)
    
    #Objective:
    def objective_rule(model):
        costX = 0.0
        costU = 0.0
        costTerminal = 0.0
        for t in model.tIDX:
            for i in model.uIDX:
                for j in model.uIDX:
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
        for i in model.xIDX:
            for j in model.xIDX:               
                costTerminal += (model.x[i, model.N]-x_goal[i]) * model.P[i, j] * (model.x[j, model.N]-x_goal[j])
        return costX + costU + costTerminal
    
    model.cost = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
    
    # Constraints:
    def buildDynamicsConstraints(model,i,t):
        xPlus = dynModel(model,t)
        return model.x[i,t+1] -xPlus[i] == 0.0 if t < model.N else pyo.Constraint.Skip

    model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=buildDynamicsConstraints)
    
    def buildInitialConstraints(model,i):
        return model.x[i,0]-x0[i] == 0.0  #
    
    model.init_const = pyo.Constraint(model.xIDX, rule=buildInitialConstraints)
    
    def buildMaxInputConstraints(model,i,t):
        return model.u[i,t] <= uU
    model.max_input_constraints = pyo.Constraint(model.uIDX, model.tIDX, rule=buildMaxInputConstraints)
    
    def buildMinInputConstraints(model,i,t):
        return uL <= model.u[i,t]
    model.min_input_constraints = pyo.Constraint(model.uIDX, model.tIDX, rule=buildMinInputConstraints)
        
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)
    
    if str(results.solver.termination_condition) == "optimal":
        feas = True
    else:
        feas = False
            
    xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
    uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T
    
    JOpt = model.cost()
      
    return [model, feas, xOpt, uOpt, JOpt]

if __name__ == "main":
    N = 200
    [model, feas, xOpt, uOpt, JOpt] = solve_cftoc(LinearDynamicsModel, 0*np.eye(1),np.eye(2), N, np.array([0,0]),np.array([np.pi,0]), 1, -1, 1)

    time_vec = np.arange(N+1)*dt

    # simple example
    plt.plot(time_vec, xOpt[0, :], 'b-')
    plt.xlabel('t')
    plt.ylabel('x1')
    plt.axis('equal')
    plt.show()
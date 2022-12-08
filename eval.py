import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulator_data_gen import generate_dataset
from dataset import SimDataset
from utils import torchify
from simulator_data_gen import *
from analytic_mpc import solve_cftoc, LinearDynamicsModel
from heuristic_mpc import MPCPolicyBaseline


def DynamicsAdapter(mlp, histories, numGDSteps, baseline=False):
    """
    histories: (T, 7)
    """
    if len(histories) > 2:
        # Get inputs
        take_last_n = 50
        if len(histories) >= take_last_n + 2:
            inputs = torchify(histories[-take_last_n-1:-2, :], device=device)    
            targets = torchify(histories[-take_last_n:-1, [1]], device=device)
        else:
            inputs = torchify(histories[0:-2, :], device=device)    
            targets = torchify(histories[1:-1, [1]], device=device)

        loss_function = nn.MSELoss()
        for epoch in range(0, numGDSteps):
            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)
            loss.backward()

            # Perform optimization
            # optimizer.step()
            # scheduler.step()
    
    inputCurrentVec = torchify(histories[[-1], :], device=device)
    if baseline:
        return mlp(inputCurrentVec)
    else:
        f = mlp.forward_f(inputCurrentVec)
        g = mlp.forward_g(inputCurrentVec)
        return f,g

def MPC(dynModel, numGDSteps, R,P, N, M, x0 ,xg, nx, nu, uL, uU, mlp, mpc_baseline, baseline=False):

    uOpt = np.zeros((nu, M))
    xReal = np.zeros((nx,M+1))
    xReal[:,0] = x0.reshape(nx, )
    xMeas = np.zeros((nx,M+1))
    xMeas[0,0], xMeas[1,0] = statesMeasured(xReal[0,0],xReal[1,0])
    xPred = np.zeros((nx, N+1, M))

    feas = np.zeros((M, ), dtype=bool)
    fL_k,fU_k = fLimits(xMeas[0,0], xMeas[1,0])
    gL_k,gU_k = gLimits(xMeas[0,0], xMeas[1,0])
    fBL_k,fBU_k = fBaselineLimits(xMeas[0,0], xMeas[1,0])
    histories = [[xMeas[0,0], xMeas[1,0], 0, fU_k, fL_k, gU_k, gL_k, fBU_k, fBL_k]]
    loss_history = []
    
    for t in range(M):
        print(t)
        if baseline:
            u, x = mpc_baseline.get_action(np.array(histories, dtype=np.float64)[-1, :2])
            x_predict = DynamicsAdapter(mpc_baseline.dyn_models[0], np.array(histories, dtype=np.float64), numGDSteps, baseline=True)
            uActual = u[0]
        else:
            [model, feas[t], x, u, J] = solve_cftoc(dynModel, R, P, N, xMeas[:, t], xg, nu, uL, uU)
            if not feas[t]:
                xOpt = []
                uOpt = []
                break
        
            f,g = DynamicsAdapter(mlp, np.array(histories, dtype=np.float64), numGDSteps)
            uActual = (xMeas[1,t] + u[:,0]*dt - f.detach().cpu().numpy()) / g.detach().cpu().numpy()
            
        xp1,wp1 = dynamicsReal(xReal[0, t],xReal[1, t],uActual)  
        
        if baseline:
            # print("predicted vs real for x:", x[0, 0], xp1)
            print("predicted vs real for w:", x[0, 1], wp1)

            # print(xp1,wp1)
            xReal[0,t+1],xReal[1,t+1] = xp1, wp1 
            xp2,wp2 =  statesMeasured(xp1,wp1)
            xMeas[0,t+1],xMeas[1,t+1] = xp2, wp2
            uOpt[:, t] = uActual.reshape(nu, )
            
            fL_k,fU_k = fLimits(xp2,wp2)
            gL_k,gU_k = gLimits(xp2,wp2)
            fBL_k,fBU_k = fBaselineLimits(xp2,wp2)
            histories[-1][2] = uActual
            histories.append([xp2,wp2, 0, fU_k, fL_k, gU_k, gL_k, fBU_k, fBL_k])
            loss_history.append(x[0, 1] - wp1[0])
        else:
            # print("predicted vs real for x:", x[0, 1], xp1)
            print("predicted vs real for w:", x[1, 1], wp1)

            xReal[0,t+1],xReal[1,t+1] = xp1, wp1 
            xp2,wp2 =  statesMeasured(xp1,wp1)
            xMeas[0,t+1],xMeas[1,t+1] = xp2, wp2
            uOpt[:, t] = uActual.reshape(nu, )
            
            fL_k,fU_k = fLimits(xp2,wp2)
            gL_k,gU_k = gLimits(xp2,wp2)
            fBL_k,fBU_k = fBaselineLimits(xp2,wp2)
            histories[-1][2] = uActual
            histories.append([xp2,wp2, 0, fU_k, fL_k, gU_k, gL_k, fBU_k, fBL_k])
            loss_history.append(x[1, 1] - wp1[0])
        

    return xReal, xMeas, uOpt, loss_history



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PATH = "./"
    
    nx = 2
    nu = 1
    N = 50
    R = 0*np.eye(1)
    P = np.eye(2)
    uL = -20
    uU = 20
    x0 = np.array([0,0])
    xg = np.array([np.pi,0])
    M = 300   # Simulation time
    numGDSteps = 20

    
    mlp = DynamicsModel()
    mlp.load_state_dict(torch.load(PATH + "mlp.pt"))
    mlp_baseline = DynamicsModelBaseline()
    mlp_baseline.load_state_dict(torch.load(PATH + "mlp_baseline.pt"))
    mlp_bounded_baseline = DynamicsModelBoundedBaseline()
    mlp_bounded_baseline.load_state_dict(torch.load(PATH + "mlp_bounded_baseline.pt"))
    mlp_unbounded = DynamicsModelUnbounded()
    mlp_unbounded.load_state_dict(torch.load(PATH + "mlp_unbounded.pt"))
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100 * numGDSteps, gamma=0.5)
    
    mpc_baseline_with_mlp = MPCPolicyBaseline(ac_dim=1, ob_dim=2, ac_space_low=uL, ac_space_high=uU, 
                    dyn_models=[mlp], 
                    horizon=40,
                    N=500,
                    goal=xg,
                    sample_strategy='cem',
                    cem_iterations=4,
                    cem_num_elites=100,
                    cem_alpha=1)

    mpc_baseline_with_mlp_unbounded = MPCPolicyBaseline(ac_dim=1, ob_dim=2, ac_space_low=uL, ac_space_high=uU, 
                    dyn_models=[mlp_unbounded], 
                    horizon=40,
                    N=500,
                    goal=xg,
                    sample_strategy='cem',
                    cem_iterations=4,
                    cem_num_elites=100,
                    cem_alpha=1)

    mpc_baseline_with_mlp_baseline = MPCPolicyBaseline(ac_dim=1, ob_dim=2, ac_space_low=uL, ac_space_high=uU, 
                    dyn_models=[mlp_baseline], 
                    horizon=40,
                    N=500,
                    goal=xg,
                    sample_strategy='cem',
                    cem_iterations=4,
                    cem_num_elites=100,
                    cem_alpha=1)

    mpc_baseline_with_mlp_bounded_baseline = MPCPolicyBaseline(ac_dim=1, ob_dim=2, ac_space_low=uL, ac_space_high=uU, 
                    dyn_models=[mlp_bounded_baseline], 
                    horizon=40,
                    N=500,
                    goal=xg,
                    sample_strategy='cem',
                    cem_iterations=4,
                    cem_num_elites=100,
                    cem_alpha=1)

    xReal_baseline_with_mlp_baseline, xMeas_baseline_with_mlp_baseline, uOpt_baseline_with_mlp_baseline, loss_history_baseline_with_mlp_baseline = MPC(LinearDynamicsModel, numGDSteps, R,P, N, M, x0 ,xg, nx, nu, uL, uU, mlp_baseline, mpc_baseline_with_mlp_baseline, baseline=True)
    xReal_baseline_with_mlp_bounded_baseline, xMeas_baseline_with_mlp_bounded_baseline, uOpt_baseline_with_mlp_bounded_baseline, loss_history_bounded_baseline_with_mlp_baseline = MPC(LinearDynamicsModel, numGDSteps, R,P, N, M, x0 ,xg, nx, nu, uL, uU, mlp_bounded_baseline, mpc_baseline_with_mlp_bounded_baseline, baseline=True)
    xReal_baseline_with_mlp, xMeas_baseline_with_mlp, uOpt_baseline_with_mlp, loss_history_baseline_with_mlp = MPC(LinearDynamicsModel, numGDSteps, R,P, N, M, x0 ,xg, nx, nu, uL, uU, mlp, mpc_baseline_with_mlp, baseline=True)
    xReal_baseline_unboounded, xMeas_baseline_unboounded, uOpt_baseline_unboounded, loss_history_baseline_unboounded = MPC(LinearDynamicsModel, numGDSteps, R,P, N, M, x0 ,xg, nx, nu, uL, uU, mlp_unbounded, mpc_baseline_with_mlp_unbounded, baseline=True)
    xReal_unbounded, xMeas_unbounded, uOpt_unbounded, loss_history_unbounded = MPC(LinearDynamicsModel, numGDSteps, R,P, N, M, x0 ,xg, nx, nu, uL, uU, mlp_unbounded, None, baseline=False)
    xReal, xMeas, uOpt, loss_history = MPC(LinearDynamicsModel, numGDSteps, R,P, N, M, x0 ,xg, nx, nu, uL, uU, mlp, None, baseline=False)

    #print('Feasibility =', feas)    
    time_vec = np.arange(M+1)*dt

    
    # Plot prediction error during execution 
    plt.plot(np.abs(loss_history_baseline_with_mlp_baseline), label= 'W/O Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(np.abs(loss_history_bounded_baseline_with_mlp_baseline), label= 'W/ Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(np.abs(loss_history_baseline_with_mlp), label='W/ Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(np.abs(loss_history_baseline_unboounded), label='W/O Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(np.abs(loss_history_unbounded), label='W/O Bounds, W/ Decomposition, Analytic MPC')
    plt.plot(np.abs(loss_history), label='W/ Bounds, W/ Decomposition, Analytic MPC (Ours)')
    plt.ylim(-0.01, 0.16)
    plt.legend()
    plt.xlabel('t', fontsize=16)
    plt.ylabel('Dynamics Prediction Error (L1)', fontsize=14)
    plt.grid()
    plt.savefig(osp.join(PATH, "execution_error.png"), dpi=400, bbox_inches='tight')
    plt.show()
    
    # Plot state trajectory (position) during execution
    plt.plot(time_vec, xReal_baseline_with_mlp_bounded_baseline[0, :], label='W/O Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_baseline_with_mlp_baseline[0, :], label='W/ Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_baseline_with_mlp[0, :], label='W/ Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_baseline_unboounded[0, :], label='W/O Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_unbounded[0, :], label='W/O Bounds, W/ Decomposition, Analytic MPC')
    plt.plot(time_vec, xReal[0, :], label='W/ Bounds, W/ Decomposition, Analytic MPC (Ours)')
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x1', fontsize=16)
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.savefig(osp.join(PATH, "execution_x1.png"), dpi=400, bbox_inches='tight')
    plt.show()
    
    # Plot state trajectory (velocity) during execution
    plt.plot(time_vec, xReal_baseline_with_mlp_bounded_baseline[1, :], label='W/O Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_baseline_with_mlp_baseline[1, :], label='W/ Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_baseline_with_mlp[1, :], label='W/ Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_baseline_unboounded[1, :], label='W/O Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(time_vec, xReal_unbounded[1, :], label='W/O Bounds, W/ Decomposition, Analytic MPC')
    plt.plot(time_vec, xReal[1, :], label='W/ Bounds, W/ Decomposition, Analytic MPC (Ours)')
    plt.xlabel('t', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.savefig(osp.join(PATH, "execution_x2.png"), dpi=400, bbox_inches='tight')
    plt.show()
    
    # Plot x1 vs x2 during execution
    plt.plot(xReal_baseline_with_mlp_bounded_baseline[0, :], xReal_baseline_with_mlp_bounded_baseline[1, :], label='W/O Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(xReal_baseline_with_mlp_baseline[0, :], xReal_baseline_with_mlp_baseline[1, :], label='W/ Bounds, W/O Decomposition, Heuristic MPC')
    plt.plot(xReal_baseline_with_mlp[0, :], xReal_baseline_with_mlp[1, :], label='W/ Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(xReal_baseline_unboounded[0, :], xReal_baseline_unboounded[1, :], label='W/O Bounds, W/ Decomposition, Heuristic MPC')
    plt.plot(xReal_unbounded[0, :], xReal_unbounded[1, :], label='W/O Bounds, W/ Decomposition, Analytic MPC')
    plt.plot(xReal[0, :], xReal[1, :], label='W/ Bounds, W/ Decomposition, Analytic MPC (Ours)')
    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.ylim(-0.1, 7)
    # plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.savefig(osp.join(PATH, "execution_x1_x2.png"), dpi=400, bbox_inches='tight')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulator_data_gen import generate_dataset
from dataset import SimDataset



def train(mlp, print_g = False):
    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4)
    loss_history = []
    g_part_history = []
    # Run the training loop
    for epoch in range(0, 2500): # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        indices = np.arange(int(len(dataset)*0.8))
        inputs, targets = dataset.__getitem__(indices)

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        loss_history.append(current_loss)
        print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss))
        current_loss = 0.0
        if print_g:
            g_part = mlp.forward_g(inputs)
            print(g_part)
            g_part_history.append(g_part.mean().detach().cpu().numpy())
    # Process is complete.
    print('Training process has finished.')

    return mlp, loss_history, g_part_history



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PATH = "./"
    
    nTimesteps = np.array([2]*10000) # 10000 trajectories of length 2
    xk,wk,xkp1,wkp1,uk,fL_k,fU_k,gL_k,gU_k,fBL_k,fBU_k = generate_dataset(nTimesteps)
    
    dataset = SimDataset(xk, wk, uk, fU_k, fL_k, gU_k, gL_k, fBU_k, fBL_k, xkp1, wkp1)
    mlp = DynamicsModel(); mlp.to(device)
    mlp_baseline = DynamicsModelBaseline(); mlp_baseline.to(device)
    mlp_bounded_baseline = DynamicsModelBoundedBaseline(); mlp_bounded_baseline.to(device)
    mlp_unbounded = DynamicsModelUnbounded(); mlp_unbounded.to(device)
    
    mlp, loss_history_mlp, g_part_history_mlp = train(mlp, print_g = True)
    torch.save(mlp.state_dict(), osp.join(PATH), "mlp.pt")
    
    mlp_baseline, loss_history_mlp_baseline, g_part_history_mlp_baseline = train(mlp_baseline, print_g = False)
    torch.save(mlp_baseline.state_dict(), osp.join(PATH), "mlp_baseline.pt")
    
    mlp_bounded_baseline, loss_history_mlp_bounded_baseline, g_part_history_mlp_bounded_baseline = train(mlp_bounded_baseline, print_g = False)
    torch.save(mlp_bounded_baseline.state_dict(), osp.join(PATH), "mlp_bounded_baseline.pt")
    
    mlp_unbounded, loss_history_mlp_unbounded, g_part_history_mlp_unbounded = train(mlp_unbounded, print_g = True)
    torch.save(mlp_unbounded.state_dict(), osp.join(PATH), "mlp_unbounded.pt")
    
    # Plot the loss history
    plt.plot(loss_history_mlp_baseline[:500], label='W/O Bounds, W/O Decomposition')
    plt.plot(loss_history_mlp_unbounded[:500], label='W/O Bounds, W/ Decomposition')
    plt.plot(loss_history_mlp_bounded_baseline[:500], label='W/ Bounds, W/O Decomposition')
    plt.plot(loss_history_mlp[:500], label='W/ Bounds, W/ Decomposition (Ours)')
    plt.ylabel('MSE loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(osp.join(PATH, "training_loss.png"), dpi=400, bbox_inches='tight')
    plt.show()
    
    # Plot the learned h(.) history
    plt.plot(g_part_history_mlp_unbounded[:500], label='W/O Bounds, W/ Decomposition')
    plt.plot(g_part_history_mlp[:500], label='W/ Bounds, W/ Decomposition (Ours)')
    plt.ylabel('Learned h(.)', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    # plt.axhline(dt/J, label="Ground Truth", color='black')
    plt.legend(fontsize=12)
    plt.grid()
    plt.savefig(osp.join(PATH, "convergence_h.png"), dpi=400, bbox_inches='tight')
    plt.show()
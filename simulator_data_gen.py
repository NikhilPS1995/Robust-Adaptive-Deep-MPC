import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from simulator_constants import *


def frictionReal(x,w):
    '''
    LuGre Model for dry fiction
    
    This is the function that needs to be learnt
    We can make this function more complicated or 
    less complicated depending on how good our MLP is !    
    
    This function takes in the angular velocity as the input
    and outputs the value of friction at that instant.
    
    Friction only depends on the velocity, But Normal force could depend on position !!!
    
    '''
    NormForce = m*l*w**2/2-m*g*np.cos(x)
    Fs = mu_s*NormForce
    Fd = mu_d*NormForce
    dryFriction = -(Fd+(Fs-Fd)*np.exp(-(abs(w)*l)/vs)**gamma)*np.sign(w) # Quasi Static Lugre Model
    hydrodynamicDrag = -c*w*abs(w)

    netResistance = dryFriction*d+hydrodynamicDrag*l
    
    # F = -Fd*np.tanh(w*l/vs); # Smooth Coloumb Friction model
    
    # F = (-Fd-(Fs-Fd)*np.exp(-c*abs(w)))*np.sign(w) # Benson exponential friction model       
    
    
    return netResistance


def frictionLimits(x,w):
    
    N_max = m_max*l_max*w**2/2+m_max*g_max
    Fs_max = mu_s*N_max
    D_max = c_max*abs(w)**2

    fU = Fs_max*d+D_max*l_max
    fL = -fU

    return fL,fU

def fLimits(x,w):
    fricL,fricU = frictionLimits(x,w)
    fL = w+(fricL-m_max*g_max*l_max/2*np.sin(x))*dt/J_max
    fU = w+(fricU+m_max*g_max*l_max/2*np.sin(x))*dt/J_min
    return fL,fU 

def gLimits(x,w):
    gU = dt/J_min
    gL = dt/J_max
    return gL,gU 

def fBaselineLimits(x,w,uMax=31): # Change this number from 31 if you are changing the training inputs
    fL,fU = fLimits(x,w)
    gL,gU = gLimits(x,w)
    fBaselineL = fL-gU*uMax
    fBaselineU = fL+gU*uMax
    
    return fBaselineL, fBaselineU

def dynamicsReal(x,w,u):
    '''
    The dynamics model takes in the current state of the system 
    i.e, the current angular position x, the current angular velocity w and the input torque u.
    The output xPlus and wPlus are the states at the next time step
    
    I am using a simple euler discretization to obtain the discrete time model 
    of the continuous dynamics. this should be good enough as the sample time is very small.
    '''
    xPlus = x + w*dt
    netResistance = frictionReal(x,w)
    netTorque = netResistance + u - mgl/2*np.sin(x)
    wPlus = netTorque*dt/J+w
    
    return xPlus, wPlus



def statesMeasured(x,w):

    std_dev_x = 0.01
    std_dev_w = 0.01
    xm = x+0*np.random.normal(0,std_dev_x,x.shape)
    wm = w+0*np.random.normal(0,std_dev_w,w.shape)   

    return xm,wm 

def generateTrajectories(nTimesteps):
    '''
    nTimesteps is a list containing natural numbers. 
    Its length is the number of trajectories we want.
    ith element is the number of time steps we want in the ith trajectory.
    '''
    #Initializing the trajectories as empty cells
    xTrajs = {}
    wTrajs = {}  
    uTrajs = {}
    tTrajs = {}
    
    # number of trajectories
    nTrajs = len(nTimesteps) 
    
    
    for i in range(nTrajs): # building ith trajectory
        
        # initialize the trajectories to be zeros and later populate
        xTrajs[str(i)] = np.zeros((nTimesteps[i],1)) 
        wTrajs[str(i)] = np.zeros((nTimesteps[i],1))
        uTrajs[str(i)] = np.zeros((nTimesteps[i]-1,1))
        
        # time steps for plotting
        tTrajs[str(i)] = np.arange(0,(nTimesteps[i])*dt,dt) 
        
        # Initial conditions of the system - randomly generated using normal distribution
        xTrajs[str(i)][0] = np.random.uniform(-5,5) #random initial position
        wTrajs[str(i)][0] = np.random.uniform(-5,5) #random initial velocity
        
        
        for j in range(nTimesteps[i]-1):
            
            #input torque to the system (later this would be the output of an MPC controller)
            # for now its just a sinusoid, can be any vector
            uTrajs[str(i)][j] = 0*J * np.sin( (i+1) * j * dt ) +np.random.uniform(-30,30)
            
            
            # Build the time trajectories from the discrete time model
            # we can add noise to the outputs of the system if we want as a next step
            xTrajs[str(i)][j+1],wTrajs[str(i)][j+1] = dynamicsReal( xTrajs[str(i)][j]  ,  wTrajs[str(i)][j] , uTrajs[str(i)][j]) 

            # Use this if you want add noise to the measurements
            xTrajs[str(i)][j+1],wTrajs[str(i)][j+1] = statesMeasured(xTrajs[str(i)][j+1],wTrajs[str(i)][j+1])           
            
    return xTrajs, wTrajs, uTrajs, tTrajs

def generate_dataset(nTimesteps):
    xTrajs, wTrajs, uTrajs, tTrajs = generateTrajectories(nTimesteps)
    
    nDataPoints = np.sum(nTimesteps)-len(nTimesteps)
    count = 0
    xk       = np.zeros((nDataPoints,1))
    wk       = np.zeros((nDataPoints,1))
    xkp1     = np.zeros((nDataPoints,1))
    wkp1     = np.zeros((nDataPoints,1))
    uk       = np.zeros((nDataPoints,1))
    fL_k     = np.zeros((nDataPoints,1))
    fU_k     = np.zeros((nDataPoints,1))
    gL_k     = np.zeros((nDataPoints,1))
    gU_k     = np.zeros((nDataPoints,1))
    fBL_k     = np.zeros((nDataPoints,1)) # baseline
    fBU_k     = np.zeros((nDataPoints,1))


    for i in range(len(wTrajs)):
        for j in range(len(wTrajs[str(i)])-1):
            uk[count][0] = uTrajs[str(i)][j]
            xk[count][0] = xTrajs[str(i)][j]
            wk[count][0] = wTrajs[str(i)][j]
            xkp1[count][0] = xTrajs[str(i)][j+1]
            wkp1[count][0] = wTrajs[str(i)][j+1]
            
            fL_k[count][0],fU_k[count][0] = fLimits(xTrajs[str(i)][j],wTrajs[str(i)][j])
            gL_k[count][0],gU_k[count][0] = gLimits(xTrajs[str(i)][j],wTrajs[str(i)][j])
            fBL_k[count][0],fBU_k[count][0] = fBaselineLimits(xTrajs[str(i)][j],wTrajs[str(i)][j])
            
            count+= 1
    return xk,wk,xkp1,wkp1,uk,fL_k,fU_k,gL_k,gU_k,fBL_k,fBU_k

fLimits_vec = np.vectorize(fLimits)
gLimits_vec = np.vectorize(gLimits)
fBaselineLimits_vec = np.vectorize(fBaselineLimits)

if __name__ == '__main__':
    if os.path.exists("Data.csv"):
        os.remove("Data.csv")
    
    nTimesteps = np.array([2]*10000) # number of times steps for simulation 

    xk,wk,xkp1,wkp1,uk,fL_k,fU_k,gL_k,gU_k,fBL_k,fBU_k = generate_dataset(nTimesteps)
            
    dataframe = pd.DataFrame(list(zip(xk,wk,uk,xkp1,wkp1,fL_k,fU_k,gL_k,gU_k,fBL_k,fBU_k)),
                             columns = ['xk','wk','uk','xkp1','wkp1','fL_k','fU_k','gL_k','gU_k','fBL_k','fBU_k'])        

    dataframe.to_csv("Data.csv")

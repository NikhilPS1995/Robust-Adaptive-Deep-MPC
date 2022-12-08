# Robust-Adaptive-Deep-MPC

## Requirement:
In the root directory of the repo, run 
```
pip install -q pyomo
wget -N -q "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"
unzip -o -q ipopt-linux64
```
Additionally, make sure `numpy`, `pandas`, `matplotlib`, and `torch` are installed.

## Code Structure
``data`` contains the generated offline data for training the dynamics model. It is not used by ``train.py`` but can be made so easily.

``dynamic_models`` implement the proposed neural network architecture (i.e., with bounds and with decomposition) and various ablations.

``mpc`` implements the analytic-based MPC based on ipopt as well as the heuristic-based MPC based on random shooting and CEM.

``simulator`` implements the dynamics for a single-link robot manipulator under the dry friction model.

``trained_models`` contains the weights for all trained models and figures.

``utils`` contains some utility functions.

``train.py`` and ``eval.py`` are the two core scripts for running training and evaluation respectively.

## To Run the Code
```
python simulator/simulator_data_gen.py
```
This generates the dataset, saved in the `data` folder.

```
python train.py
```
`train.py` generates the training data using code in the ``simulator`` folder, and trains the proposed model and baseline models using the generated data, and saves the model weights in the `trained_models` folder. Figures such as the training loss curves are also saved there.

```
python eval.py
```
``eval.py`` loads the trained models, and evaluates the performance on the simulator with different MPC controller, as implemented in the ``mpc`` folder.
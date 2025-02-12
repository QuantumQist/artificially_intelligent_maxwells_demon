# artificially_intelligent_maxwells_demon

Source code used to generate the results in arXiv:2408.15328

This repository contains three folders, each corresponding to a distinct set of reinforcement learning (RL) models used to generate the results in the paper. The folders are:

- **`discrete_measurement`** – RL agents implementing policies based on discrete measurements with fixed qubit gaps.  
- **`continuous_measurement`** – RL agents implementing policies based on continuous measurements.  
- **`projective_measurement`** – RL agents implementing policies based on projective measurements.  

## Folder Contents

**`discrete_measurement`** and - **`continuous_measurement`** include the following components:

- **`evaluate_agent.ipynb`** – A Jupyter notebook for testing policies produced by trained RL agents.  
- **`train_agent.py`** – A script for training RL agents. After training, the agent is saved in the `data` folder in the main directory.
- **Other essential files** that define the RL agents and environments. Modification of these files is not recommended.  
  - RL environments are defined in files prefixed with `sac_tri_envs`.  
  - Core agent functionalities are implemented in files prefixed with `core` and `sac_tri`.  
- **`data` folder** – Contains the trained RL agents for the corresponding models. Users can conveniently interact with these agents in the `evaluate_agent.ipynb` notebook.

**`projective_measurement`** includes the following components:

- **`src`** – A folder containing the RL code, the RL environment, and helper functions for plotting.
- **`jupyter`** – A folder containing Jupyter Notebook used to train, evaluate and plot the results.
- **`jupyter/qubit_feedback_bosonic_demon_pow_diss_training.ipynb`** – A Jupyter Notebook to train the two-level system in the thermalization dominated regime, including power and dissipation tradeoffs.
- **`jupyter/produce_pareto_data.ipynb`** – Once an RL agent is trained using the previous Jupyter Notebooks, one can evaluate and export its performance using this Jupyter Notebook.
- **`jupyter/paper_plots.ipynb`** – Code used to generate the corresponding plots in the manuscript.

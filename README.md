# artificially_intelligent_maxwells_demon

Source code used to generate the results in arXiv:2408.15328

This repository contains two folders, each corresponding to a distinct set of reinforcement learning (RL) models used to generate the results in the paper. The folders are:

- **`discrete_measurement`** – RL agents implementing policies based on discrete measurements with fixed qubit gaps.  
- **`continuous_measurement`** – RL agents implementing policies based on continuous measurements.  

## Folder Contents

Each folder includes the following components:

- **`evaluate_agent.ipynb`** – A Jupyter notebook for testing policies produced by trained RL agents.  
- **`train_agent.py`** – A script for training RL agents. After training, the agent is saved in the `data` folder in the main directory.
- **Other essential files** that define the RL agents and environments. Modification of these files is not recommended.  
  - RL environments are defined in files prefixed with `sac_tri_envs`.  
  - Core agent functionalities are implemented in files prefixed with `core` and `sac_tri`.  
- **`data` folder** – Contains the trained RL agents for the corresponding models. Users can conveniently interact with these agents in the `evaluate_agent.ipynb` notebook.

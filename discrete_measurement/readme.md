# Discrete Measurement RL Agents and Environments

This folder contains the RL agents and environments used to generate results for policies based on discrete measurements.  

## 1. Playing with the Trained Agents  

The `evaluate_agent.ipynb` notebook allows users to visualize the policies produced by the trained agents.  

## 2. Training the Agents  

The `train_agent.py` script enables training of an RL agent.  

The environments are defined in the `sac_tri_envs_dis.py` file. Modification of this file is not recommended. The following classes can be used during training:  

- **`sac_tri_envs_dis.TwoLevelDemonDisPowDiss`** – RL agent constrained to $\sigma_X$ measurement.  
- **`sac_tri_envs_dis.TwoLevelDemonDisPowDiss2`** – RL agent constrained to $\sigma_Z$ measurement.  
- **`sac_tri_envs_dis.TwoLevelDemonDisPowDissTheta`** – RL agent with an adaptive measurement angle.  

## 3. Trained Agents Used in the Paper  

The `data` folder contains the trained RL agents used to generate the results presented in the paper.  

# Continuous Measurement RL Agents and Environments  

This folder contains the RL agents and environments used to generate results for policies based on continuous measurements.  

## 1. Playing with the Trained Agents  

The `evaluate_agent.ipynb` notebook allows users to visualize the policies produced by the trained agents.  

## 2. Training the Agents  

The `train_agent.py` script is used to train RL agents.  

The environments are defined in the `sac_tri_envs_con.py` file. Modification of this file is not recommended. The following classes can be used for training:  

- **`sac_tri_envs_con.TwoLevelDemonConPowDissX`** – RL agent constrained to $\sigma_X$ measurement with a fixed qubit gap.  
- **`sac_tri_envs_con.TwoLevelDemonConPowDissZ`** – RL agent constrained to $\sigma_Z$ measurement with a fixed qubit gap.  
- **`sac_tri_envs_con.TwoLevelDemonConPowDissTheta`** – RL agent with an adaptive measurement angle and a fixed qubit gap.  
- **`sac_tri_envs_con.TwoLevelBosonicFeedbackDemonPowDissContMeas`** – RL agent constrained to $\sigma_Z$ measurement with an adaptive qubit gap.  

## 3. Trained Agents Used in the Paper  

The `data` folder contains the trained RL agents used to generate the results presented in the paper.  

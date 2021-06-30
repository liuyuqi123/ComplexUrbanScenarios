#  Complex Urban Scenarios for Autonomous Driving

## 1 Introduction
This repository contains codes for the Complex Urban Scenarios for Autonomous Driving under intersection scenarios.
In this repository, we deploy a set of autonomous agents developed based on Reinforcement learning(RL) methods.



methods for training testing respectively.
In our research work, we use CARLA as the autonomous driving simulator.


## 2 Get started
### 2.0 system requirements
Ubuntu 16.04-18.04  
CARLA 0.9.10.1 and above


### 2.1 set-up environment
Anaconda is suggested for python environment management. 
First of all, create the conda env with  
`conda env create -f gym_carla.yml -n gym_carla`
then a conda env named with gym_carla will be created, 
the conda env can be activated with command  
`conda activate gym_carla`

Before you do anything, please make sure the carla client is running correctly,
following carla commands are suggested:
`./CarlaUE4.sh -opengl -quality-level=Low -ResX=400 -ResY=300 -carla-rpc-port=2000`
Carla port number is alternative, however you should make sure it coordinates with the value in the codes.


### 2.1 training
Codes for training can be found in 
`./train/rl_agents/td3/single_task_without_attention`

Run the training procedure by running `rl_agent.py`  
In CMD, Using  
`python rl_agent.py --help`  
to learn the usage of available parameters.

During the training procedure, the weights of the RL network will be saved in
``

We provide a baseline agent with trained weights stored in  
`` 
You can replace the weight file with your own weight dict.

### 2.2 testing
Before running the test code, please fix your repository path  
``

We deploy a set of traffic scenarios for testing the trained agent.
If you wish to test your trained agent, please put your tensorflow NN weights in   
``


## 3 Intersection Scenarios
In this part, a series of scenarios of urban intersection are deployed to validate the
performance of the RL agent.
We develop traffic scenarios using CARLA scenario_runner module.

## 4 Baselines & Results 
In our work, we deploy a td3 agent to handle the intersection turning task.
Here are the testing results of our proposed traffic scenarios.

### Sceario 1: Left turning with oppose continuous traffic flow turning left


### Sceario 2: Left turning with oppose continuous traffic flow turning left


## Citation

[comment]: <> (@misc{lia_corrales_2015_15991,)

[comment]: <> (    author       = {Lia Corrales},)

[comment]: <> (    title        = {{dust: Calculate the intensity of dust scattering halos in the X-ray}},)

[comment]: <> (    month        = mar,)

[comment]: <> (    year         = 2015,)

[comment]: <> (    doi          = {10.5281/zenodo.15991},)

[comment]: <> (    version      = {1.0},)

[comment]: <> (    publisher    = {Zenodo},)

[comment]: <> (    url          = {https://doi.org/10.5281/zenodo.15991})

[comment]: <> (    })


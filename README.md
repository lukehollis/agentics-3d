# Unity Agentics - RL-Agents Based Character AI System


This project is a work-in-progress implementing world model training for Unity environments using the ML-Agents to OpenAI Gym wrapper to build realistic human simulations. The goal is to train reinforcement learning agents efficiently by having them learn in a world model rather than directly in the environment so that they have a persistent inner state, and then use their policy to make decisions in the environment.

![Rome Simulator](https://iiif.mused.com/rome_simulator_mused.jpg/0,240,2048,854/990,/0/default.jpg)

This is currently running the [Civilization Simulations](https://mused.com/explore/simulations/), simulating human history. You can also implement it easily with the [Happy Harvest](https://assetstore.unity.com/packages/essentials/tutorial-projects/happy-harvest-2d-sample-project-259218) 2d template from Unity.

![BART Digital Twin](https://iiif.mused.com/digital_twin_bart.jpg/0,240,2048,854/990,/0/default.jpg)

Now the package is also functional in 3D, used in the BART Digital Twin simulation project. [link soon](#)

![roman_farm_simulator_neural_state_800w](https://github.com/user-attachments/assets/bd9e2a5e-8593-4f58-bbcc-a33e8d300aed)
Individual characters can be controlled by policy and run inference on shader graph, then visualize their inner state in game.

## Overview

I'm currently developing this to build digital twins of real world systems, modern and historical, but it can be used for training Generalist Agents for simulations for other purposes. 

The project implements three key components based on modern world model approaches:

1. **Data Collection**: Gather experience from Unity environments using ML-Agents Gym wrapper
2. **World Model Training**: Learn to predict next observations and rewards
3. **Policy Training**: Train RL agents inside the learned world model

Then the policy can be used in a Unity environment for controlling NPC behavior or other purposes.


## Getting Started

### Prerequisites
- Python 3.10+
- Unity ML-Agents 
- PyTorch
- OpenAI Gym

### Installation
```bash
pip install -r requirements.txt
``` 

### Data Collection
```bash
python src/collect.py
```

## Training

The training process follows two main stages that can be run with a single command:


```bash
python src/train.py 
```

This will automatically:

1. **Train World Model**: Updates the world model on collected experiences from the Unity environment to better predict next observations, rewards, and episode terminations. The world model combines a VAE for compact state representation with an MDN-RNN for dynamics prediction.

2. **Train Policy in Imagination**: Optimizes the agent's policy entirely inside the learned world model using actor-critic RL. This allows rapid policy improvement without additional environment interaction.



# In Unity

## Getting Started
0. Add the Agentics package to your Unity project
1. Add the Brain component to your character
2. Configure required components (Sensor, Plan, Motivation, etc.)
3. Set up training configuration with the python directory in the root of this repo
4. Add example plans in Data directory and configure with NetworkingController [coming soon]


# Citations

```
@incollection{ha2018worldmodels,
  title = {Recurrent World Models Facilitate Policy Evolution},
  author = {Ha, David and Schmidhuber, J{\"u}rgen},
  booktitle = {Advances in Neural Information Processing Systems 31},
  pages = {2451--2463},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution},
  note = "\url{https://worldmodels.github.io}",
}
```


```
@inproceedings{Park2023GenerativeAgents,  
author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},  
title = {Generative Agents: Interactive Simulacra of Human Behavior},  
year = {2023},  
publisher = {Association for Computing Machinery},  
address = {New York, NY, USA},  
booktitle = {In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},  
keywords = {Human-AI interaction, agents, generative AI, large language models},  
location = {San Francisco, CA, USA},  
series = {UIST '23}
}
```

```
@inproceedings{alonso2024diffusionworldmodelingvisual,
      title={Diffusion for World Modeling: Visual Details Matter in Atari},
      author={Eloi Alonso and Adam Jelley and Vincent Micheli and Anssi Kanervisto and Amos Storkey and Tim Pearce and François Fleuret},
      booktitle={Thirty-eighth Conference on Neural Information Processing Systems}}
      year={2024},
      url={https://arxiv.org/abs/2405.12399},
}
```

```
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}   
```


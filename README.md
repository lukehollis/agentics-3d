# Unity Agentics - RL-Agents Based Character AI System


This project implements world model training for Unity environments using the ML-Agents to OpenAI Gym wrapper. The goal is to train reinforcement learning agents efficiently by having them learn in a learned world model rather than directly in the environment.

This is currently running the [Civilization Simulations](https://mused.com/explore/simulations/)



## Overview

World models allow agents to:
- Learn environment dynamics from raw observations
- Train policies efficiently in a learned simulation
- Transfer learned behaviors back to the real environment

We implement three key components based on modern world model approaches:

1. **Data Collection**: Gather experience from Unity environments using ML-Agents Gym wrapper
2. **World Model Training**: Learn to predict next observations and rewards
3. **Policy Training**: Train RL agents inside the learned world model

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
python src/collect_data.py
```

### World Model Training
```bash
python src/train_world_model.py
```

### Policy Training
```bash
python src/train_policy.py
```

## Implementation Details

The data collection pipeline follows best practices from recent world model papers:

- Observations are preprocessed to 64x64 pixels
- Data is collected in sequences for temporal prediction
- Separate train/validation splits are maintained
- Both random and policy-driven data collection supported

## License

MIT

# In Unity

## Core Components
- AgentBrain: Core ML-Agents implementation
- AgentSensor: Environmental perception
- AgentPlanSystem: Task and goal management
- MotivationSystem: Emotional and needs simulation
- ConsciousnessSystem: Internal state and thought processing
- WorldModelInference: Predictive world modeling

## Getting Started
0. Add the Agentics package to your Unity project
1. Add the AgentBrain component to your character
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
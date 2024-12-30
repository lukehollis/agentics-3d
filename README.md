# Unity Agentics - RL-Agents Based Character AI System


This project is a work-in-progress implementing generalist agent training for Unity environments using the ML-Agents package to build realistic simulations. The goal is to be able to more easily have agents available for simulations for a wide range of purposes, especially in urban and transportation simulations.

![Rome Simulator](https://iiif.mused.com/rome_simulator_mused.jpg/0,240,2048,854/990,/0/default.jpg)

This is currently running the [Civilization Simulations](https://mused.com/explore/simulations/), simulating human history. You can also implement it easily with the [Happy Harvest](https://assetstore.unity.com/packages/essentials/tutorial-projects/happy-harvest-2d-sample-project-259218) 2d template from Unity.

![BART Digital Twin](https://iiif.mused.com/digital_twin_bart.jpg/0,240,2048,854/990,/0/default.jpg)

Now the package is also functional in 3D, used in the BART Digital Twin simulation project. [link soon](#)


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
0.a -- if working in 2D, you may get further with the 2D specific version 
1. Add the `AgenticController` component to your character, and configure with CharacterController, NavMeshAgent, and any other components you need
2. Set an initial day plan and waypoints for the agent
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

```
@article{sima2024,
    title={SIMA: A Generalist AI Agent for 3D Virtual Environments},
    author={SIMA Team},
    journal={Google DeepMind Blog},
    year={2024},
    month={March},
    url={https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/}
}
```



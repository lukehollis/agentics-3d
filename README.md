


https://github.com/user-attachments/assets/d730fd1c-5b08-4699-ad8f-e1a0e8aa44e2


# Generalist Character AI System for Simulating Social Situations


This project is a work-in-progress implementing generalist agent character training for Unity environments using the ML-Agents package to build realistic simulations. The goal is to be able to more easily have agents available for simulations for a wide range of purposes, especially in urban and transportation simulations.

![Rome Simulator](https://iiif.mused.com/rome_simulator_mused.jpg/0,240,2048,854/800,/0/default.jpg)

This is currently running the [Civilization Simulations](https://mused.com/explore/simulations/), simulating human history. You can also implement it easily with the [Happy Harvest](https://assetstore.unity.com/packages/essentials/tutorial-projects/happy-harvest-2d-sample-project-259218) 2d template from Unity.

![BART Digital Twin](https://iiif.mused.com/digital_twin_bart.jpg/0,240,2048,854/800,/0/default.jpg)

Now the package is also functional in 2D and 3D, used in the BART Digital Twin simulation project. [link soon](#)

Individual characters can be controlled by policy, running inference in engine, and interpret and plan actions with a language model, then visualize their inner state in game.

## Cognitive Architecture

![cognitive_architecture](https://github.com/user-attachments/assets/ae749a95-908a-4321-b750-5f5ee11df80a)

Inspired by the [CoALA Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427), the NPCs in the simulations build on a similar version of the language agents but only implement the language model for rational reflection on actions inferred from policy--so a "impulse" from the policy and then considered rational reflection from the language model. 

https://github.com/user-attachments/assets/c42531ca-64e2-4ea9-92a5-a44fab5b04c9


## Instatiating Characters in Environments

You can generate a diverse population of characters representing your required demographic landscape based on realworld data inputs. The character creation process synthesizes data from multiple sources, including US Census Bureau demographics, Bureau of Labor Statistics employment data, and local transportation studies.

Each generated character includes contextual details including their personality, age, profession, education level, commute patterns, family relationships, backstory (generalized and with episodic highlights), and other information. 

![NPC Spawn](https://iiif.mused.com/bart_simulation_spawning_npcs.jpg/full/800,/0/default.jpg)

The Unity NPC Spawner Tool provided is an Editor tool that easily allows you to customize your NPC creation process on your navmesh within a bounding box. 

The generation process begins by establishing family units, then populating them with individual characters whose attributes are derived from weighted distributions matching specified input demographics. Educational backgrounds influence personality assignments, while employment roles and industries are selected to mirror the region's actual workforce composition.

## Mode shift

The Mode Shift module simulates how individual agents make transportation choices in urban environments. Each agent uses a multi-factor decision model to choose between available transportation modes (car, public transit, walking, cycling) based on:

* Travel time and distance
* Cost considerations
* Weather conditions
* Individual agent preferences
* Time of day
* Current traffic conditions
* Transit service availability



https://github.com/user-attachments/assets/ae6daae7-9fb0-4321-ac94-e252f89cb40c



### Implementation
Agents use a combination of:

* Reinforcement learning policy for immediate reactions to environment changes
* Language model reflection for longer-term transportation planning
* Historical behavior patterns that influence future choices



## SEIR Simulation

The SEIR (Susceptible, Exposed, Infectious, Recovered) module integrates epidemiological modeling with agent-based simulation to model disease spread in urban environments. This integration allows for realistic modeling of how transportation patterns and urban mobility affect disease transmission.

https://github.com/user-attachments/assets/38f2ad40-e490-4d00-9eb2-269f6b6e0596


## Networked State Management System

Inspired by projects like Photon for Unity networked state management, the state management system orchestrates persistent offline networked states through interconnected components in Unity and the backend (Python/Django connected via websocket) that work together to create a cohesive game world.

The Django repo for the backend is at https://github.com/lukehollis/agentics-backend, to be public soon -- if you want access, just request.

#### Conversations
The conversation system tracks all character-player interactions by maintaining a detailed message history with timestamps. It supports different types of communication (SPATIAL, DIGITAL) and maintains visibility settings to control information flow between characters and players. Each conversation is tied to specific game sessions and users, allowing for contextual interactions that persist across sessions.

If time-period appropriate, characters have the option to use simulated digital means of communication that are a simplified version of our own. They can do 1-1 agent dialog, group communication, or public posting on "The Feed," similar to a social network. Other characters can choose to check the Feed as one of their behaviors and like/dialog with public posts. 


#### Memory System
Characters maintain memories through a hierarchical storage system with three priority levels: high, medium, and low. Currently in dialog, the most recent ten high priority memories, five medium priority memories, and last three background memories provide general context. Each memory record contains a description, location, timestamp, priority level (ranked separately with a language model), and associations with specific characters, users, and game sessions.


https://github.com/user-attachments/assets/68d52aa0-2de4-4895-8844-9ac24141851f


#### World State
The world state maintains comprehensive data about the game environment, including the time period, environmental conditions, and the relations and data of characters. The backend contains a basic represetnation of spatial information in the Unity 2d or 3d environments while managing the progression of quests and storylines. This creates a persistent environment for characters -- which responds to player and character actions between play sessions.



https://github.com/user-attachments/assets/6887b20e-4f93-4ed8-85f2-05cdb81efdd1


## Active Development 

This project is in active development in public, and I'll update it here for anyone that's interested, but because I release code separately from the 3d art, it does not yet contain a complete functioning standalone project. If you're interested in using the Google Maps photorealistic geotiles, I recommend checking out the [Cesium project](https://github.com/CesiumGS/) for your desired engine. 


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
0.a If working in 2D, you may get further with the 2D specific version in this repo
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
@misc{sumers2023cognitive,
      title={Cognitive Architectures for Language Agents}, 
      author={Theodore Sumers and Shunyu Yao and Karthik Narasimhan and Thomas L. Griffiths},
      year={2023},
      eprint={2309.02427},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
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



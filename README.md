# lunar-lander: Implementation of (double) deep Q-learning for training an agent to play the game lunar lander

## Introduction

In this repository we implement an agent that is trained to play the game lunar lander using (double) deep Q-learning. Here is a video of a trained agent playing the game:

https://user-images.githubusercontent.com/37583039/219359191-7988e0dc-b1a4-43cc-82d1-4cc18be0d0a2.mp4

We use the lunar lander implementation from [gymnasium](https://gymnasium.farama.org). For the implementation of deep Q-learning we follow <a href="#ref_1">Ref. [1]</a>, for the implementation of double deep Q-learning we follow <a href="#ref_2">Ref. [2]</a>.

In the following, we first list the files contained in this repository and explain their usage. We then show a comparison of agent training and post-training performance of agents trained using deep Q-learning vs. double deep Q-learning.

## Files and usage

* [agent_class.py](https://github.com/juliankappler/lunar-lander/blob/main/agent_class.py): In this python file we implement the agent class which, which we use both for training an agent and acting with the trained agent
* [train and visualize agent.ipynb](https://github.com/juliankappler/lunar-lander/blob/main/train%20and%20visualize%20agent.ipynb): In this Jupyter notebook we train an agent, and subsequently create a gameplay video. The video at the beginning of this readme file was created with this notebook
* [train_agent.py](https://github.com/juliankappler/lunar-lander/blob/main/train_agent.py): In this python script we train an agent, and save both the trained agent parameters, as well as its training statistics to the disk

## Comparison: deep Q-learning vs. double deep Q-learning


## References

<a id="ref_1">[1] **Playing Atari with Deep Reinforcement Learning**. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602).</a>

<a id="ref_2">[2] **Deep Reinforcement Learning with Double Q-learning**. Hado van Hasselt, Arthur Guez, David Silver. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461).</a>

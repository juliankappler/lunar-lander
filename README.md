# lunar-lander: Implementation of (double) deep Q-learning for training an agent to play the game lunar lander

## Introduction

In this repository we implement an agent that is trained to play the game lunar lander using (double) deep Q-learning. Here is a video of a trained agent playing the game:

https://user-images.githubusercontent.com/37583039/219359191-7988e0dc-b1a4-43cc-82d1-4cc18be0d0a2.mp4

We use the lunar lander implementation from [gymnasium](https://gymnasium.farama.org). For the implementation of deep Q-learning we follow <a href="#ref_1">Ref. [1]</a>, for the implementation of double deep Q-learning we follow <a href="#ref_2">Ref. [2]</a>.

In the following, we [first](#files-and-usage) list the files contained in this repository and explain their usage. We [then](#comparison-deep-q-learning-vs-double-deep-q-learning) compare the training speed and post-training performance of agents trained using deep Q-learning vs. double deep Q-learning, and hard vs. soft update.

## Files and usage

* [agent_class.py](agent_class.py): In this python file we implement the agent class which, which we use both for training an agent and acting with the trained agent
* [train and visualize agent.ipynb](train%20and%20visualize%20agent.ipynb): In this Jupyter notebook we train an agent, and subsequently create a gameplay video. The video at the beginning of this readme file was created with this notebook
* [train_agent.py](train_agent.py): In this python script we train an agent, and save both the trained agent parameters, as well as its training statistics to the disk
* [run_agent.py](train_agent.py): In this python script we run episodes for an already trained agent, and save statistics (duration of episodes, return for each episode) to the disk
* [trained_agents/batch_train_and_run.sh](trained_agents/batch_train_and_run.sh): With this bash script we train 500 agents (via [train_agent.py](train_agent.py)) and subsequently run 1000 episodes for each trained agent (via [run_agent.py](run_agent.py)). The script by default runs 10 processes in parallel.
* [trained_agents/plot_results.ipynb](trained_agents/plot_results.ipynb): In this Jupyter notebook we analyze the training statistics and performance of the trained agents from [batch_train_and_run.sh](trained_agents/batch_train_and_run.sh), as summarized in [this section](#comparison-deep-q-learning-vs-double-deep-q-learning).

## Comparison: deep Q-learning vs. double deep Q-learning, and soft update vs. hard update

With the script [batch_train_and_run.sh](trained_agents/batch_train_and_run.sh) we first train 500 agents and then run 1000 episodes for each agent for the following three scenarios:
1. Deep q-learning (DQN) with hard update, meaning that the target net is always set equal to the policy net after each training epoch.
2. Deep q-learning (DQN) with soft update with update parameter $\tau = 0.01$.
3. Double deep q-learning (dDQN) with soft update with update parameter $\tau = 0.01$.

Here is a plot showing the distribution of the episodes needed for training for each scenario, along with the mean:

![training_n_episodes](https://user-images.githubusercontent.com/37583039/220933734-8c995abf-b963-473b-9c06-7488191c19c9.png)

We observe that the training for the two scenarios with soft update proceeds faster than for the DQN algorithm with hard update. Note that for the DQN algorithm with hard update, 7 of the 500 trainings did not fulfill our stopping criterion within 10000 environment episodes of collecting training data and were thus stopped as unsuccessful (1.4% failure rate). The plot above only considers the 493 successful training instances.

For each successfully trained agent, we run 1000 episodes. Here is a plot of the resulting distribution of all episode returns for each scenario:

![return_distribution](https://user-images.githubusercontent.com/37583039/220934293-83d7cd52-0cac-4344-b105-29859d8ecaf1.png)

We observe that the distributions are all very similar, with some minor differences between the hard-update DQN and the other two algorithms. 

We conclude that using soft update for the target net instead of a hard update improves the speed of training. For soft update, both DQN and dDQN lead to comparable training speed. All training scenarios considered lead to comparable agents (as measured by their return).

## References

<a id="ref_1">[1] **Playing Atari with Deep Reinforcement Learning**. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602).</a>

<a id="ref_2">[2] **Deep Reinforcement Learning with Double Q-learning**. Hado van Hasselt, Arthur Guez, David Silver. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461).</a>

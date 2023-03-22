# lunar-lander: Reinforcement learning algorithms for training an agent to play the game lunar lander

## Introduction

In this repository we implement an agent that is trained to play the game lunar lander using <i>i)</i> an actor-critic algorithm, and <i>ii)</i> a (double) deep Q-learning algorithm. Here is a video of a trained agent playing the game:

https://user-images.githubusercontent.com/37583039/219359191-7988e0dc-b1a4-43cc-82d1-4cc18be0d0a2.mp4

We use the lunar lander implementation from [gymnasium](https://gymnasium.farama.org). For the implementation of the actor-critic algorithm we loosely follow <a href="#ref_1">Ref. [1]</a>. While for the implementation of deep Q-learning we follow <a href="#ref_2">Ref. [2]</a>, for the implementation of double deep Q-learning we follow <a href="#ref_3">Ref. [3]</a>.

In the following, we [first](#files-and-usage) list the files contained in this repository and explain their usage. We [then](#comparison-actor-critic-algorithm-vs-deep-q-learning) compare the training speed and post-training performance of agents trained using the actor-critic algorithm and deep Q-learning.

## Files and usage

* [agent_class.py](agent_class.py): In this python file we implement the agent class which, which we use both for training an agent and acting with the trained agent
* [train and visualize agent.ipynb](train%20and%20visualize%20agent.ipynb): In this Jupyter notebook we train an agent, and subsequently create a gameplay video. The video at the beginning of this readme file was created with this notebook
* [train_agent.py](train_agent.py): In this python script we train an agent, and save both the trained agent parameters, as well as its training statistics to the disk
* [run_agent.py](train_agent.py): In this python script we run episodes for an already trained agent, and save statistics (duration of episodes, return for each episode) to the disk
* [trained_agents/batch_train_and_run.sh](trained_agents/batch_train_and_run.sh): With this bash script we train 500 agents (via [train_agent.py](train_agent.py)) and subsequently run 1000 episodes for each trained agent (via [run_agent.py](run_agent.py)). The script by default runs 8 processes in parallel.
* [trained_agents/plot_results.ipynb](trained_agents/plot_results.ipynb): In this Jupyter notebook we analyze the training statistics and performance of the trained agents from [batch_train_and_run.sh](trained_agents/batch_train_and_run.sh), as summarized in [this section](#comparison-actor-critic-algorithm-vs-deep-q-learning).

## Comparison: actor-critic algorithm vs. deep Q-learning

With the script [batch_train_and_run.sh](trained_agents/batch_train_and_run.sh) we first train 500 agents and then run 1000 episodes for each agent using
1. the actor-critic algorithm, and
2. the deep q-learning (DQN) algorithm.

Here is a plot showing the distribution of the episodes needed for training for each scenario, along with the mean:

![training_n_episodes](https://user-images.githubusercontent.com/37583039/227004864-4bc5a4f4-6df3-4edd-a389-af3dbdf8da92.png)

We observe that the distribution of episodes needed for training is more spread out for the actor-critic method. Furthermore, the actor-critic algorithm on average needed 28% more episodes to complete the training as compared to the DQN algorithm.

Here is a plot showing the actual runtime distribution of the respective 500 trainings:

![training_execution_time](https://user-images.githubusercontent.com/37583039/227005536-58dad8cc-1cd4-4c0f-a2c3-53b1f3d6a0fd.png)

On average, the actor-critic algorithm takes 67% longer to train as compared to deep Q-learning. Note that in the actor-critic algorithm we have twice as many parameters as compared to the Q-learning algorithm. This is because all neural networks we use are of equal size, and in the actor-critic algorithm we train 2 neural networks (namely the actor and the critic).

For each trained agent, we run 1000 episodes. Here is a plot of the resulting distribution of all episode returns for each scenario:

![return_distribution](https://user-images.githubusercontent.com/37583039/227030052-69404955-da7d-4a52-b778-7d3638166308.png)

We observe that the distributions are rather similar. While the DQN agents perform slightly better on average (return of 227.4 for DQN vs 211.6 for actor-critic), the support of the agent-critic distributions returns extends a bit further to the right (high returns) of the plot.

To investigate this last point further, we for both algorithms select the agent that yielded the highest average return in its 1000 episodes, and plot the respective return distribution:

![return_distribution_best](https://user-images.githubusercontent.com/37583039/227030006-bbb71677-1c2e-4369-8b53-1fc0a7b1f5ac.png)

We see that the best actor-critic agent performs slightly better than the best DQN agent (mean return 261.9 vs. mean return 238.0). 

We summarize:
- Compared to the actor-critic algorithm, the DQN algorithm yielded both a smaller mean and variance for the number of training episodes necessary for successful training. On average, the DQN algorithm also needed less time for training.
- The mean return of the trained DQN agents was slightly larger as compared to the actor-critic agents.
- However, the single best DQN agent performed a little bit worse than the single best actor-critic agent.

So overall, while the DQN algorithm on average trains faster and yields better mean performance, in our samples the best agent was obtained from the actor-critic algorithm.

In a more thorough study one might want to increase the number of trained agents to see whether our best actor being an actor-critic agent was a random fluctuation. Furthermore, one might want to vary the hyperparameters for training, so as to optimize the number of training episodes for each algorithm.

## References

<a id="ref_1">[1] **Reinforcement Learning: An Introduction**. Richard S. Sutton, Andrew G. Barto. [http://incompleteideas.net/book/the-book.html](http://incompleteideas.net/book/the-book.html).</a>

<a id="ref_2">[2] **Playing Atari with Deep Reinforcement Learning**. Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602).</a>

<a id="ref_3">[3] **Deep Reinforcement Learning with Double Q-learning**. Hado van Hasselt, Arthur Guez, David Silver. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461).</a>

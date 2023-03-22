#!/usr/bin/env python

import argparse
import os
import time
import gymnasium as gym

import agent_class as agent
# agent_class defines the following classes:
# - "neural_network"
# - "memory"
# - "agent_base" with derived classes
#   - "dqn" for (double) deep Q-learning
#   - "actor_critic" for the actor-critic learning algorithm

############################################################################
# This script trains an agent to play the gym environment "LunarLander-v2" #
############################################################################

# Example for run from command line:
#
#          python train_agent.py --f my_file --verbose --overwrite
#
# This command will train the neural network and
# - save both snapshots from the training and 
#   the final network as a dictionary to my_file.tar,
# - save training stats (such as return for each episode during training) to
#   my_file_training_data.h5
# - save training runtime to 
#   my_file_execution_time.txt
# Because of the flag "--verbose", the training progress will be printed to
# the terminal (default is False, so that nothing will be printed to the 
# terminal)
# Because of the flag "--overwrite", any possibly existing files with the
# names of the current output files will be overwritten (default is False,
# and the program stops if it finds existing training files)
#
# Note that further below in this script, all the training-algorithm 
# independent parameters are set to their default values.

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str, default='my_agent',
                    help='output filename (suffix will be added by script)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--dqn', action='store_true') # use this flag to train 
                                                # via deep Q-learning
parser.add_argument('--ddqn', action='store_true') # use this flag to train 
                                                # via double deep Q-learning
parser.set_defaults(dqn=False)
parser.set_defaults(ddqn=False)
args = parser.parse_args()

# Create output filenames
output_filename = '{0}.tar'.format(args.f)
output_filename_training_data = '{0}_training_data.h5'.format(args.f)
output_filename_time = '{0}_execution_time.txt'.format(args.f)
verbose=args.verbose
overwrite=args.overwrite
dqn=args.dqn
ddqn=args.ddqn

if not overwrite:
    # Comment the following out if you want to overwrite
    # existing model/training data
    error_msg = ("File {0} already exists. If you want to overwrite"
        " that file, please restart the script with the flag --overwrite.")
    if os.path.exists(output_filename):
        raise RuntimeError(error_msg.format(output_filename))
    if os.path.exists(output_filename_training_data):
        raise RuntimeError(error_msg.format(output_filename_training_data))

# Create environment
env = gym.make('LunarLander-v2')

# Obtain dimensions of action and observation space
N_actions = env.action_space.n
observation, info = env.reset()
N_state = len(observation)
if verbose:
    print('dimension of state space =',N_state)
    print('number of actions =',N_actions)

# Set parameters
# NOTE: Only the first two parameters (N_state and N_actions) are mandatory, 
# the reminaing parameters are optional.
# For demonstration, we here set all algorithm-independent optional parameters
# to their default. Because for all the extra parameters below we use their
# default values, using
#      parameters = {'N_state':N_state, 'N_actions':N_actions}
# instead of the dictionary below will yield the same results.
#
parameters = {
    # Mandatory parameters
    'N_state':N_state,
    'N_actions':N_actions,
    #
    # All the following parameters are optional, and we set them to
    # their default values here:
    #
    'discount_factor':0.99, # discount factor for Bellman equation
    #
    'N_memory':20000, # number of past transitions stored in memory
                        # for experience replay
    #
    # Optimizer parameters
    'training_stride':5, # number of simulation timesteps between
        # optimization (learning) steps
    'batch_size':32, # mini-batch size for optimizer
    'saving_stride':100, # every saving_stride episodes, the 
        # current status of the training is saved to disk
    #
    # Parameters for stopping criterion for training
    'n_episodes_max':10000, # maximal number of episodes until the 
        # training is stopped (if stopping criterion is not met before)
    'n_solving_episodes':20, # the last N_solving episodes need to 
        # fulfill both:
    # i) minimal return over last N_solving_episodes:
    'solving_threshold_min':200.,
    # ii) mean return over last N_solving_episodes:
    'solving_threshold_mean':230.,
        }

# Instantiate agent class
if dqn or ddqn:
    if ddqn:
        parameters['doubledqn'] = True
    #
    my_agent = agent.dqn(parameters=parameters)
else:
    my_agent = agent.actor_critic(parameters=parameters)


# Train agent on environment
start_time = time.time()
training_results = my_agent.train(
                        environment=env,
                        verbose=verbose,
                        model_filename=output_filename,
                        training_filename=output_filename_training_data,
                            )
execution_time = (time.time() - start_time)
with open(output_filename_time,'w') as f:
    f.write(str(execution_time))

if verbose:
    print('Execution time in seconds: ' + str(execution_time))



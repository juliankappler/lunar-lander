#!/usr/bin/env python

import argparse
import os
import gymnasium as gym

from agent_class import *
# agent_class defines
# - class "neural_network"
# - class "agent"
# - function "train_agent"

############################################################################
# This script trains an agent to play the gym environment "LunarLander-v2" #
############################################################################

# Example for run from command line:
#
#          python train_model.py --f my_file --verbose --overwrite
#
# This command will train the neural network and
# - save both snapshots from the training and 
#   the final network as a dictionary to my_file.tar,
# - save training stats (such as return for each episode during training) to
#   my_file_training_data.h5
# Because of the flag "--verbose", the training progress will be printed to
# the terminal (default is False, so that nothing will be printed to the 
# terminal)
# Because of the flag "--overwrite", any possibly existing files with the
# names of the current output files will be overwritten (default is False,
# and the program stops if it finds existing training files)
#
# Note that all the training parameters are set further below in this script.
# (If desired, one could easily add some training parameters to the parser.)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str, default='my_agent',
                    help='output filename (suffix will be added by script)')
parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--overwrite', action='store_true')
parser.set_defaults(overwrite=False)
args = parser.parse_args()

# Create output filenames
output_filename = '{0}.tar'.format(args.f)
output_filename_training_data = '{0}_training_data.h5'.format(args.f)
verbose=args.verbose
overwrite=args.overwrite

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
# For demonstration, we here set all optional parameters and use the standard
# values that would have been used if he had not provided those parameters.
# Phrased differently: Using 
#      parameters = {'N_state':N_state, 'N_actions':N_actions}
# instead of the dictionary below will yield the exact same results, because 
# for all the extra parameters below we use the standard values the program 
# would use anyway
#
parameters = {
    # Mandatory parameters
    'N_state':N_state,
    'N_actions':N_actions,
    # neural network topoology (number of neurons at each hidden layer)
    'layers':[64,32],
    # Use deep Q-learning (DQL) or double deep Q-learning (dDQL)?
    'doubleDQN':False, # False -> use DQL; True -> use dDQL
    #
    'discount_factor':0.99, # discount factor for Bellman equation
    #
    'N_memory':200000, # number of past transitions stored in memory
                        # for experience replay
    #
    # Optimizer & loss function parameters
    'learning_rate':1e-3,
    'training_stride':5, # number of simulation timesteps between
        # optimization (learning) steps
    'batch_size':32, # mini-batch size for optimizer
    'loss_choice':'MSELoss',
    'optimizer_choice':'RMSprop',
    'saving_stride':100, # every saving_stride episodes, the 
        # current status of the training is saved to disk
    #
    'target_net_update_stride':1, # soft update stride for target net
    'target_net_update_tau':1., # soft update parameter for target net
    #
    # Parameters for epsilon-greedy policy with epoch-dependent epsilon
    'epsilon':1.0, # initial value for epsilon
    'epsilon_1':0.1, # final value for epsilon
    'd_epsilon':0.00005, # decrease of epsilon
        # after each training epoch
    #
    # Parameters for stopping criterion for training
    'N_episodes_max':10000, # maximal number of episodes until the 
        # training is stopped (if stopping criterion is not met before)
    'N_solving_episodes':20, # the last N_solving episodes need to 
        # fulfill both:
    # i) minimal return over last N_solving_episodes:
    'solving_threshold_min':230.,
    # ii) mean return over last N_solving_episodes:
    'solving_threshold_mean':250.,
        }
    
# Instantiate agent class
my_agent = agent(parameters=parameters)

# Train agent on environment
training_results = my_agent.train(
                        environment=env,
                        verbose=verbose,
                        model_filename=output_filename,
                        training_filename=output_filename_training_data,
                            )



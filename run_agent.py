#!/usr/bin/env python

import gymnasium as gym
import os
import argparse
import torch
import numpy as np
import itertools 
import h5py 

import agent_class as agent
# agent_class defines
# - class "neural_network"
# - class "agent"
# - function "train_agent"


#############################################################################
# This code runs the gym environment "LunarLander-v2" using a trained agent #
#############################################################################

# Example for run from command line:
#
#   python run_agent.py --f my_file --verbose --overwrite --N 500
#
# This command loads the agent state stored in my_file.tar, uses this agent
# to run N = 500 episodes of the LunarLander-v2 environment, and stores the
# resulting list of returns per episode and duration per episode in the file
# my_file_trajs.h5
# 
# Because of the flag "--verbose", the simulation progress is printed to
# the terminal. Default is False, so that nothing is printed.
# Because of the flag "--overwrite", a possibly existing file my_file_trajs.h5
# will be overwritten. Default is False, and the program stops if it finds an
# existing file my_file_trajs.h5.
#

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str, default='my_agent',
        help='input/output filename (suffix will be added by script)')
parser.add_argument('--N',type=int, default=1000,
        help='number of simulations')
parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--overwrite', action='store_true')
parser.set_defaults(overwrite=False)
parser.add_argument('--dqn', action='store_true') # use this flag to train 
                                                # via deep Q-learning
parser.add_argument('--ddqn', action='store_true') # use this flag to train 
                                                # via double deep Q-learning
parser.set_defaults(dqn=False)
parser.set_defaults(ddqn=False)
args = parser.parse_args()

# Create input and output filenames
input_filename = '{0}.tar'.format(args.f)
output_filename = '{0}_trajs.tar'.format(args.f)
N = args.N
verbose=args.verbose
overwrite=args.overwrite
dqn=args.dqn
ddqn=args.ddqn
if ddqn:
    dqn = True

if not overwrite:
    # Comment the following out if you want to overwrite
    # existing model/training data
    error_msg = ("File {0} already exists. If you want to overwrite"
        " that file, please restart the script with the flag --overwrite.")
    if os.path.exists(output_filename):
            raise RuntimeError(error_msg.format(output_filename))

def run_and_save_simulations(env, # environment
                            input_filename,output_filename,N=1000,
                            dqn=False):
    #
    # load trained model
    input_dictionary = torch.load(open(input_filename,'rb'))
    dict_keys = np.array(list(input_dictionary.keys())).astype(int)
    max_index = np.max(dict_keys)
    input_dictionary = input_dictionary[max_index] # During training we 
    # periodically store the state of the neural networks. We now use
    # the latest state (i.e. the one with the largest episode number), as 
    # for any succesful training this is the state that passed the stopping
    # criterion
    #
    # instantiate agent
    parameters = input_dictionary['parameters']
    # Instantiate agent class
    if dqn:
        my_agent = agent.dqn(parameters=parameters)
    else:
        my_agent = agent.actor_critic(parameters=parameters)
    my_agent.load_state(state=input_dictionary)
    #
    # instantiate environment
    env = gym.make('LunarLander-v2')
    #
    durations = []
    returns = []
    status_string = ("Run {0} of {1} completed with return {2:<5.1f}. Mean "
            "return over all episodes so far = {3:<6.1f}            ")
    # run simulations
    for i in range(N):
        # reset environment, duration, and reward
        state, info = env.reset()
        episode_return = 0.
        #
        for n in itertools.count():
            #
            action = my_agent.act(state)
            #
            state, step_reward, terminated, truncated, info = env.step(action)
            #
            done = terminated or truncated
            episode_return += step_reward
            #
            if done:
                #
                durations.append(n+1)
                returns.append(episode_return)
                #
                if verbose:
                    if i < N-1:
                        end ='\r'
                    else:
                        end = '\n'
                    print(status_string.format(i+1,N,episode_return,
                                        np.mean(np.array(returns))),
                                    end=end)
                break
    #
    dictionary = {'returns':np.array(returns),
                'durations':np.array(durations),
                'input_file':input_filename,
                'N':N}
        
    with h5py.File(output_filename, 'w') as hf:
        for key, value in dictionary.items():
            hf.create_dataset(str(key), 
                data=value)
    

# Create environment
env = gym.make('LunarLander-v2')

run_and_save_simulations(env=env,
                            input_filename=input_filename,
                            output_filename=output_filename,
                            N=N,
                            dqn=dqn)

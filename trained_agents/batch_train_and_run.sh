#!/bin/bash

####################################################################
# Trains agents to maximize the reward in an environment, and      #
# then uses the trained agents to run episodes in that environment #
####################################################################
#
# This script trains 500 agents to play the gym environment LunarLander-v2,
# and runs 1000 episodes of the environment for each trained agent.
# By default 10 trainings/runs are run in parallel, and both the final models,
# training statistics, and statistics of the runs are saved to ./data/
n = 500 # number of agents
parallelism=10 # number of trainings/runs that are run in parallel
output_dir=$(pwd)"/data" # output dir
# 
# The script assumes that the the agent class agent_class.py and the scripts
# train_agent.py, run_agent.py are located here:
agent_dir='../' 
#
#
###############
# Basic usage #
###############
# 
# By running 
#      bash batch_train_and_run.sh
# the script generates the subdir, performs the trainings, and runs episodes
# using trained agents.
#
# If you only want to train the agents, or only want to run episodes on 
# already trained agents, you can do this by running
#      bash batch_train_and_run.sh --train-only
#      bash batch_train_and_run.sh --run-only
#
#
####################
# Training options #
####################
#
# There are two options you can set for the training algorithm via flags:
# 1. By default, the agent is trained via deep Q-learning. You can train the
#    agent via double deep Q-learning by running the script with flag
#      bash batch_train_and_run.sh --ddqn
# 2. By default, the target net of the (double) deep Q-learning algorithm is 
#    updated via a soft update with tau = 0.01. You can use a hard update for
#    the target net (meaning that the target net is set to the policy net 
#    after each training epoch) by running the script with flag 
#      bash batch_train_and_run.sh --no-soft
#    
#
##########
# Output #
##########
#
# Naming scheme for the output files for the i-th agent are as follows:
#
# 1. For deep Q-learning with soft update:
#    - parameters of trained model: agent_$i.tar 
#    - training statistics: agent_$i_training_data.h5 
#    - episode runs statistics: agent_$i_trajs.tar 
# 2. For deep Q-learning with hard update:
#    - parameters of trained model: agent_$i_no-soft.tar 
#    - training statistics: agent_$i_no-soft_training_data.h5 
#    - episode runs statistics: agent_$i_no-soft_trajs.tar 
# 3. For double deep Q-learning with soft update:
#    - parameters of trained model: agent_$i_ddqn.tar 
#    - training statistics: agent_$i_ddqn_training_data.h5 
#    - episode runs statistics: agent_$i_ddqn_trajs.tar 
#
#
###################
# Note on runtime #
################### 
#
# On my computer, training a single agent on a single CPU takes about 5-30 
# minutes (depending on algorithm/whether hard and soft updates are used). 
# Thus, an estimate for training 500 agents with 10 CPUs in parallel is
#
#           20 min * 500 / 10 = 1000 min = 16 hours and 40 min.
#


# create output directory
mkdir -p "${output_dir}"

# Switch to directory where we execute train_agent.py
cd "$agent_dir"

# argument parser, adapted from https://linuxcommand.org/lc3_wss0120.php
flags_train=""
flags_run=""
filename="agent"
train=true
run=true
while [ "$1" != "" ]; do
  case $1 in
    -d | --ddqn )
      flags_train="${flags_train} --ddqn"
      filename="${filename}_ddqn"
      ;;
    --no-soft )
      flags_train="${flags_train} --no-soft"
      filename="${filename}_no-soft"
      ;;
    -t | --train-only | --no-run )
      run=false
      ;;
    -r | --run-only | --no-train )
      train=false
      ;;
    -o | --overwrite )
      flags_train="${flags_train} --overwrite"
      flags_run="${flags_run} --overwrite"
      ;;
    * )
      echo "unknown command line argument ${1}"
      ;;
  esac
  shift
done


# This function is called for each agent in the for loop below
run_command () {
  local i=$1
  #
  if [ "$train" = true ] ; then
    echo $(date +"%Y-%m-%d %T") " Training agent ${filename}_${i}"
    # Run the command with the current suffix in the filename
    python train_agent.py --f "${output_dir}/${filename}_${i}" $flags_train
  fi
  #
  if [ "$run" = true ] ; then
    echo $(date +"%Y-%m-%d %T") " Running agent ${filename}_${i}"
    # Run the command with the current suffix in the filename
    python run_agent.py --f "${output_dir}/${filename}_${i}" $flags_run
  fi
}

# Loop over the number of trainings and run the commands in parallel
for ((i=1; i <= $n; i++));
  # Wait until there are less than $parallelism commands running
  while (( $(jobs -r -p | wc -l) >= $parallelism )); do
    sleep 1
  done
  #
  run_command "$i" &
  #
done

# Wait for all background commands to finish before exiting
wait
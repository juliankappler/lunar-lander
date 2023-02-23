#!/usr/bin/env python

import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
import h5py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'mps'
import warnings

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class neural_network(nn.Module):
    '''
    Feedforward neural network with variable number
    of hidden layers and ReLU nonlinearites
    '''

    def __init__(self,
                layers=[64,32],# layers[i] = # of neurons at i-th hidden layer
                N_inputs = 4, # number of inputs
                N_outputs=2, # number of outputs
                dropout=False,
                p_dropout=0.5,
                ):
        super(neural_network,self).__init__()

        self.N_inputs = N_inputs
        self.N_outputs = N_outputs

        self.network_layers = []
        for i,neurons_in_current_layer in enumerate(layers):
            #
            if i == 0:
                self.network_layers.append(nn.Linear(self.N_inputs, 
                                                neurons_in_current_layer) )
            else:
                self.network_layers.append(nn.Linear(layers[i-1], 
                                                neurons_in_current_layer) )
            #
            if dropout:
                self.network_layers.append( nn.Dropout(p=p_dropout) )
            #
            self.network_layers.append( nn.ReLU() )
        #
        self.network_layers.append( nn.Linear(neurons_in_current_layer, 
                                                self.N_outputs)  )
        #
        self.network_layers = nn.Sequential(*self.network_layers)
        #

    def forward(self,x):
        for layer in self.network_layers:
            x = layer(x)
        return x



class agent():

    def __init__(self,parameters):
        """
        Initializes the agent class

        Keyword arguments:
        parameters -- dictionary with parameters for the agent

        There are two mandatory keys for the dictionary:
        - N_state (int): dimensionality of the (continuous) state space
        - N_actions (int): number of actions available to the agent

        All other arguments are optional, for a list see the class methods 
        set_initialization_parameters(self,parameters)
        set_parameters(self,parameters)

        """
        #
        # set parameters that are mandatory and can only be set at 
        # initializaton of a class instance
        self.set_initialization_parameters(parameters=parameters)
        #
        # set default parameters for all non-mandatory parameters
        self.set_default_parameters()
        #
        # update non-mandatory parameters according to user-provided values
        self.set_parameters(parameters=parameters)
        #
        # intialize agent neural networks 
        self.initialize_neural_networks()
        #
        # initialize the optimizer and loss function used for training
        self.initialize_optimizer()
        self.initialize_loss()


    def set_default_parameters(self):
        """Set default values for optional class parameters"""

        # Use deep Q-learning (DQL) or double deep Q-learning (dDQL)?
        self.doubleDQN = False # False -> use DQL; True -> use dDQL
        #
        self.discount_factor = 0.99 # discount factor for Bellman equation
        #
        self.N_memory = 200000 # number of past transitions stored in memory
                               # for experience replay
        self.memory = memory(self.N_memory)
        #
        # Optimizer & loss function parameters
        self.learning_rate = 1e-3
        self.training_stride = 5 # number of simulation timesteps between
            # optimization (learning) steps
        self.batch_size = 32 # mini-batch size for optimizer
        self.loss_choice = 'MSELoss'.lower()
        self.optimizer_choice = 'RMSprop'.lower()
        self.saving_stride = 100 # every saving_stride episodes, the 
            # current status of the training is saved to disk
        #
        self.target_net_update_stride = 1 # soft update stride for target net
        self.target_net_update_tau = 1e-2 # soft update parameter for target net
        #
        # Parameters for epsilon-greedy policy with epoch-dependent epsilon
        self.epsilon = 1.0 # initial value for epsilon
        self.epsilon_1 = 0.1 # final value for epsilon
        self.d_epsilon = 0.00005 # decrease of epsilon
            # after each training epoch
        #
        # Parameters for stopping criterion for training
        self.N_episodes_max = 10000 # maximal number of episodes until the 
            # training is stopped (if stopping criterion is not met before)
        self.N_solving_episodes = 20 # the last N_solving episodes need to 
            # fulfill both:
        # i) minimal return over last N_solving_episodes:
        self.solving_threshold_min = 230.
        # ii) mean return over last N_solving_episodes:
        self.solving_threshold_mean = 250.
        #

    def make_dictionary_keys_lowercase(self,dictionary):
        output_dictionary = {}
        for key, value in dictionary.items():
            output_dictionary[key.lower()] = value
        return output_dictionary

    def set_initialization_parameters(self,parameters):
        """Set those class parameters that are required at initialization"""
        #
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        try: # set mandatory parameter N_state
            self.N_state = parameters['n_state']
        except KeyError:
            raise RuntimeError("Parameter N_state (= # of input"\
                         +" nodes for neural network) needs to be supplied.")
        #
        try: # set mandatory parameter N_actions
            self.N_actions = parameters['n_actions']
        except KeyError:
            raise RuntimeError("Parameter N_actions (= # of output"\
                         +" nodes for neural network) needs to be supplied.")
        # 
        try: # set neural network architecture
            self.layers = parameters['layers']
        except KeyError:
            # default architecture: two hidden layers with 64 and 32 neurons
            self.layers = [64,32]

    def set_parameters(self,parameters):
        """Set training parameters for (double) deep Q-learning"""
        #
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        ##################################################
        # Use deep Q-learning or double deep Q-learning? #
        ##################################################
        try: # False -> use DQN; True -> use double DQN
            self.doubleDQN = parameters['doubledqn']
        except KeyError:
            pass
        #
        ########################################
        # Discount factor for Bellman equation #
        ########################################
        try: # 
            self.discount_factor = parameters['discount_factor']
        except KeyError:
            pass
        #
        #################################
        # Experience replay memory size #
        #################################
        try: # 
            self.N_memory = int(parameters['n_memory'])
            self.memory = memory(self.N_memory)
        except KeyError:
            pass
        #
        ###############################
        # Parameters for optimization #
        ###############################
        try: # learning rate for optimizer
            self.learning_rate = parameters['learning_rate']
        except KeyError:
            pass
        #
        try: # number of simulation timesteps between optimization steps
            self.training_stride = parameters['training_stride']
        except KeyError:
            pass
        #
        try: # size of mini-batch for each optimization step
            self.batch_size = int(parameters['batch_size'])
        except KeyError:
            pass
        #
        try: # loss function for optimization
            self.loss_choice = parameters['loss']
        except KeyError:
            pass
        #
        try: # optimizer
            self.optimizer_choice = parameters['optimizer']
        except KeyError:
            pass
        #
        try: # IO during training: every saving_stride episodes, the 
            # current status of the training is saved to disk
            self.saving_stride = parameters['saving_stride']
        except KeyError:
            pass
        #
        ##########################################
        # Parameters for updating the target net #
        ##########################################
        try: # after how many training epochs do we update the target net?
            self.target_net_update_stride = \
                                    parameters['target_net_update_stride']
        except KeyError:
            pass
        #
        try: # tau for soft update of target net (value 1 means hard update)
            self.target_net_update_tau = parameters['target_net_update_tau']
            # check if provided parameter is within bounds
            error_msg = ("Parameter 'target_net_update_tau' has to be "
                    "between 0 and 1, but value {0} has been passed.")
            error_msg = error_msg.format(self.target_net_update_tau)
            if self.target_net_update_tau < 0:
                raise RuntimeError(error_msg)
            elif self.target_net_update_tau > 1:
                raise RuntimeError(error_msg)
        except KeyError:
            pass
        #
        ########################################
        # Parameters for epsilon-greedy policy #
        ########################################
        try: # probability for random action for epsilon-greedy policy
            self.epsilon = \
                    parameters['epsilon']
        except KeyError:
            pass
        #
        try: # final probability for random action during training 
            #  for epsilon-greedy policy
            self.epsilon_1 = \
                    parameters['epsilon_1']
        except KeyError:
            pass
        # 
        try: # amount by which epsilon decreases during each training epoch
            #  until the final value self.epsilon_1 is reached
            self.d_epsilon = \
                    parameters['d_epsilon']
        except KeyError:
            pass
        #
        ##############################################
        # Parameters for training stopping criterion #
        ##############################################
        try: # maximal number of episodes until the training is stopped 
            # (if stopping criterion is not met before)
            self.N_episodes_max = parameters['n_episodes_max']
        except KeyError:
            pass
        #
        try: # # of the last N_solving episodes that need to fulfill the
            # stopping criterion for minimal and mean episode return
            self.N_solving_episodes = parameters['n_solving_episodes']
        except KeyError:
            pass
        #
        try: # minimal return over last N_solving_episodes
            self.solving_threshold_min = parameters['solving_threshold_min']
        except KeyError:
            pass
        #
        try: # mean return over last N_solving_episodes
            self.solving_threshold_mean = parameters['solving_threshold_mean']
        except KeyError:
            pass
        #


    def initialize_neural_networks(self):
        """Initialize policy and target networks for agent"""
        #
        self.policy_net = neural_network(N_inputs=self.N_state,
                            N_outputs=self.N_actions).to(device)
        self.target_net = neural_network(N_inputs=self.N_state,
                            N_outputs=self.N_actions).to(device)
        self.target_net.eval()
        #


    def get_number_of_model_parameters(self): 
        """Return the number of trainable neural network parameters"""
        # from https://stackoverflow.com/a/49201237
        return sum(p.numel() for p in (self.target_net).parameters() \
                                    if p.requires_grad)


    def get_parameters(self):
        """Return dictionary with parameters of the current agent instance"""

        parameters = {
        # neural network parameters
        'N_state':self.N_state,
       'N_actions':self.N_actions,
       'layers':self.layers,
       # parameters determining the basic algorithm
        'doubledqn':self.doubleDQN,
        'discount_factor':self.discount_factor,
        # memory for experience replay
        'N_memory':self.N_memory,
        # optimizer & loss function parameters
       'learning_rate':self.learning_rate,
       'training_stride':self.training_stride,
       'batch_size':self.batch_size,
       'loss_choice':self.loss_choice,
       'optimizer_choice':self.optimizer_choice,
       'saving_stride':self.saving_stride,
       # parameters for updating target net
       'target_net_update_stride':self.target_net_update_stride,
       'target_net_update_tau':self.target_net_update_tau,
       # parameters for epsilon-greedy policy with epoch-dependent epsilon
       'epsilon':self.epsilon,
       'epsilon_1':self.epsilon_1,
       'd_epsilon':self.d_epsilon,
       # parameters for stopping criterion for training
        'N_episodes_max':self.N_episodes_max,
        'N_solving_episodes':self.N_solving_episodes,
        'solving_threshold_min':self.solving_threshold_min,
       'solving_threshold_mean':self.solving_threshold_mean,
                }
        return parameters


    def get_state(self):
        '''Return dictionary with current state of neural net and optimizer'''
        #
        policy_net_state_dict = copy.deepcopy(self.policy_net.state_dict())
        target_net_state_dict = copy.deepcopy(self.target_net.state_dict())
        optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        #
        state = {'parameters':self.get_parameters(),
                'policy_net_state_dict':policy_net_state_dict,
                'target_net_state_dict':target_net_state_dict,
                 'optimizer_state_dict':optimizer_state_dict,
                }
        #
        return state


    def check_parameter_dictionary_compatibility(self,parameters):
        """Check compatibility of provided parameter dictionary with class"""

        error_string = ("Error loading state. Provided parameter {0} = {1} ",
                    "is inconsistent with agent class parameter {0} = {2}. ",
                    "Please instantiate a new agent class with parameters",
                    " matching those of the model you would like to load.")
        try: 
            N_state =  parameters['N_state']
            if N_state != self.N_state:
                raise RuntimeError(error_string.format('N_state',N_state,
                                                self.N_state))
        except KeyError:
            pass
        #
        try: 
            N_actions =  parameters['N_actions']
            if N_actions != self.N_actions:
                raise RuntimeError(error_string.format('N_actions',N_actions,
                                                self.N_actions))
        except KeyError:
            pass
        #
        try: 
            layers =  parameters['layers']
            if layers != self.layers:
                raise RuntimeError(error_string.format('layers',layers,
                                                self.layers))
        except KeyError:
            pass
        #


    def load_state(self,state):
        '''
        Load given states for neural networks and optimizer

        The argument "state" has to be a dictionary with the following 
        (key, value) pairs:

        state['parameters'] = dictionary with the agents parameters
        state['policy_net_state_dict'] = state dictionary of policy net
        state['target_net_state_dict'] = state dictionary of target net
        state['optimizer_state_dict']  = state dictionary of optimizer
        '''
        #
        parameters=state['parameters']
        #
        self.check_parameter_dictionary_compatibility(parameters=parameters)
        #
        self.set_parameters(parameters=parameters)
        #
        self.initialize_neural_networks()
        self.initialize_optimizer()
        self.initialize_loss()
        #
        self.policy_net.load_state_dict(state['policy_net_state_dict'])
        self.target_net.load_state_dict(state['policy_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])


    def initialize_optimizer(self,optimizer=None):
        """Instantiate optimizer class for policy net"""
        #
        # There must be a more elegant way to do this, in which we do not
        # need to create an if-clause for every optimizer?!
        #
        if optimizer == None:
            optimizer = self.optimizer_choice
        else:
            self.optimizer_choice = optimizer
        #
        if (self.optimizer_choice).lower() == 'Adam'.lower():
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                lr=self.learning_rate)
        elif (self.optimizer_choice).lower() == 'RMSprop'.lower():
            self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(),
                                             lr=self.learning_rate)
        elif (self.optimizer_choice).lower() == 'SGD'.lower():
            self.optimizer = torch.optim.SGD(self.policy_net.parameters(),
                                             lr=self.learning_rate)
        else:
            raise RuntimeError("Optimizer not implemented for agent.")

    def initialize_loss(self,loss=None):
        """Instantiate loss class"""
        #
        # There must be a more elegant way to do this, in which we do not
        # need to create an if-clause for every loss function?!
        #
        if loss == None:
            loss = self.loss_choice
        else:
            self.loss_choice = loss
        #
        if (self.loss_choice).lower() == 'MSELoss'.lower():
            self.loss = nn.MSELoss()
        elif (self.loss_choice).lower() == 'SmoothL1Loss'.lower():
            self.loss = nn.SmoothL1Loss()
        else:
            raise RuntimeError("Loss not implemented for agent.")

    def act(self,state,epsilon=0.):
        """
        Use policy net to select an action for the current state
        
        We use an epsilon-greedy algorithm: 
        - With probability epsilon we take a random action (uniformly drawn
          from the finite number of available actions)
        - With probability 1-epsilon we take the optimal action (as predicted
          by the policy net)

        By default epsilon = 0, which means that we actually use the greedy 
        algorithm for action selection
        """

        if torch.rand(1).item() > epsilon:
            # 
            with torch.no_grad():
                self.policy_net.eval()
                action = self.policy_net(torch.tensor(state)).argmax(0).item()
                self.policy_net.train()
                return action
        else:
            # perform random action
            return torch.randint(low=0,high=self.N_actions,size=(1,)).item()

    def add_memory(self,memory):
        """Add current experience tuple to the memory"""
        self.memory.push(*memory)

    def update_epsilon(self):
        """
        Update epsilon for epsilon-greedy algorithm
        
        For training we assume that 
        epsilon(n) = max{ epsilon_0 - d_epsilon * n ,  epsilon_1 },
        where n is the number of training epochs.

        For epsilon_0 > epsilon_1 the function epsilon(n) is piecewise linear.
        It first decreases from epsilon_0 to epsilon_1 with a slope d_epsilon,
        and then becomes constant at the value epsilon_1.
        
        This ensures that during the initial phase of training the neural 
        network explores more randomly, and in later stages of the training
        follows more the policy learned by the neural net.
        """
        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

    def run_optimization_step(self):
        """Run one optimization step for the policy net"""
        #
        # if we have less sample transitions than we would draw in an 
        # optimization step, we do nothing
        if len(self.memory) < self.batch_size:
            return
        #
        self.policy_net.train() # turn on training mode
        #
        # get random sample of transitions from memory
        current_transitions = self.memory.sample(batch_size=self.batch_size)
        #
        # convert list of Transition elements to Transition element with lists
        # (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*current_transitions))
        #
        # convert lists of current transitions to torch tensors
        state_batch = torch.cat( [s.unsqueeze(0) for s in batch.state],
                                        dim=0)#.to(device)
        # state_batch.shape = [batch_size, N_states]
        next_state_batch = torch.cat(
                         [s.unsqueeze(0) for s in batch.next_state],dim=0)
        action_batch = torch.cat(batch.action)#.to(device)
        # action_batch.shape = [batch_size]
        reward_batch = torch.cat(batch.reward)#.to(device)
        done_batch = torch.tensor(batch.done).float()#.to(device)
        #
        #
        # Evaluate left-hand side of the Bellman equation using policy net
        LHS = self.policy_net(state_batch.to(device)).gather(dim=1,
                                 index=action_batch.unsqueeze(1))
        # LHS.shape = [batch_size, 1]
        #
        # Evaluate right-hand side of Bellman equation
        if self.doubleDQN:
            # double deep-Q learning paper: https://arxiv.org/abs/1509.06461
            #
            # in double deep Q-learning, we use the policy net for choosing
            # the action on the right-hand side of the Bellman equation. We 
            # then use the target net to evaluate the Q-function on the 
            # chosen action
            argmax_next_state = self.policy_net(next_state_batch).argmax(
                                                                    dim=1)
            # argmax_next_state.shape = [batch_size]
            #
            Q_next_state = self.target_net(next_state_batch).gather(
                dim=1,index=argmax_next_state.unsqueeze(1)).squeeze(1)
            # shapes of the various tensor appearing in the previous line:
            # self.target_net(next_state_batch).shape = [batch_size,N_actions]
            # self.target_net(next_state_batch).gather(dim=1,
            #   index=argmax_next_state.unsqueeze(1)).shape = [batch_size, 1]
            # Q_next_state.shape = [batch_size]
        else:
            # in deep Q-learning, we use the target net both for choosing
            # the action on the right-hand side of the Bellman equation, and 
            # for evaluating the Q-function on that action
            Q_next_state = self.target_net(next_state_batch\
                                                ).max(1)[0].detach()
            # Q_next_state.shape = [batch_size]
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        RHS = RHS.unsqueeze(1) # RHS.shape = [batch_size, 1]
        #
        # optimize the model
        loss = self.loss(LHS, RHS)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #
        self.policy_net.eval() # turn off training mode
        #

    def soft_update_target_net(self):
        """Soft update parameters of target net"""
        #
        # the following code is from https://stackoverflow.com/q/48560227
        params1 = self.policy_net.named_parameters()
        params2 = self.target_net.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.target_net.load_state_dict(dict_params2)


    def train(self,environment,
                    verbose=True,
                    model_filename=None,
                    training_filename=None,
                ):
        """
        Train the agent on a provided environment

        Keyword arguments:
        environment -- environment used by the agent to train. This should be
                       an instance of a class with methods "reset" and "step".
                       - environment.reset() should reset the environment to
                         an initial state and return a tuple,
                            current_state, info = environment.reset(),
                         such current_state is an initial state of the with
                         np.shape(current_state) = (self.N_state,)
                       - environment.set(action) should take an integer in 
                         {0, ..., self.N_action-1} and return a tuple, 
                            s, r, te, tr, info = environment.step(action),
                         where s is the next state with shape (self.N_state,),
                         r is the current reward (a float), and where te and
                         tr are two Booleans that tell us whether the episode
                         has terminated (te == True) or has been truncated 
                         (tr == True)
        verbose (Bool) -- Print progress of training to terminal. Defaults to
                          True
        model_filename (string) -- Output filename for final trained model and
                                   periodic snapshots of the model during 
                                   training. Defaults to None, in which case
                                   nothing is not written to disk
        training_filename (string) -- Output filename for training data, 
                                      namely lists of episode durations, 
                                      episode returns, number of training 
                                      epochs, and total number of steps 
                                      simulated. Defaults to None, in which 
                                      case no training data is written to disk
        """
        #
        training_complete = False
        step_counter = 0 # total number of simulated environment steps
        epoch_counter = 0 # number of training epochs 
        #
        # lists for documenting the training
        episode_durations = [] # duration of each training episodes
        episode_rewards = [] # reward of each training episode
        steps_simulated = [] # total number of steps simulated at the end of
                             # each training episode
        training_epochs = [] # total number of training epochs at the end of 
                             # each training episode
        #
        output_state_dicts = {} # dictionary in which we will save the status
                                # of the neural networks and optimizer
                                # every self.saving_stride steps epochs during
                                # training. 
                                # We also store the final neural network
                                # resulting from our training in this 
                                # dictionary
        #
        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.N_solving_episodes))
            #
            status_progress_string = ( # for outputting status during training
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        #
        for n_episode in range(self.N_episodes_max):
            #
            # reset environment and reward of current episode
            state, info = environment.reset()
            current_total_reward = 0.
            #
            for i in itertools.count(): # timesteps of environment
                #
                # select action using policy net
                action = self.act(state=state,
                                    epsilon=self.epsilon)
                #
                # perform action
                next_state, reward, terminated, truncated, info = \
                                        environment.step(action)
                #
                step_counter += 1 # increase total steps simulated
                done = terminated or truncated # did the episode end?
                current_total_reward += reward # add current reward to total
                #
                # store the transition in memory
                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)
                self.add_memory([torch.tensor(state),
                            action,
                            torch.tensor(next_state),
                            reward,
                            done])
                #
                state = next_state
                #
                if step_counter % self.training_stride == 0:
                    # train model
                    self.run_optimization_step() # optimize
                    self.update_epsilon() # for epsilon-greedy algorithm
                    epoch_counter += 1 # increase count of optimization steps
                    if epoch_counter % self.target_net_update_stride == 0:
                        self.soft_update_target_net() # soft update target net
                #
                if done: # if current episode ended
                    #
                    # update training statistics
                    episode_durations.append(i + 1)
                    episode_rewards.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_epochs.append(epoch_counter)
                    #
                    # if we have run at least self.N_solving_episodes, check
                    # whether the stopping criterion is met
                    if n_episode > self.N_solving_episodes:
                        #
                        # calculate minimal and mean return over the last
                        # self.N_solving_episodes epsiodes 
                        recent_returns = np.array(\
                            episode_rewards[-self.N_solving_episodes:]\
                                                )
                        minimal_return = np.min(recent_returns)
                        mean_return = np.mean(recent_returns)
                        #
                        #
                        # check whether stopping criterion is met
                        if minimal_return > self.solving_threshold_min:
                            if mean_return > self.solving_threshold_mean:
                                training_complete = True
                    else:
                        minimal_return = .0
                        mean_return = .0
                    if verbose:
                            # print training stats
                            if n_episode % 100 == 0 and n_episode > 0:
                                end='\n'
                            else:
                                end='\r'
                            if minimal_return > self.solving_threshold_min:
                                if mean_return > self.solving_threshold_mean:
                                    end='\n'
                            #
                            print(status_progress_string.format(n_episode,
                                    current_total_reward,
                                   minimal_return,mean_return),
                                        end=end)
                    break
            #
            # Save model and training stats to disk
            if (n_episode % self.saving_stride == 0) \
                    or training_complete \
                    or n_episode == self.N_episodes_max-1:
                #
                if model_filename != None:
                    output_state_dicts[n_episode] = self.get_state()
                    torch.save(output_state_dicts, model_filename)
                #
                training_results = {'episode_durations':episode_durations,
                            'epsiode_returns':episode_rewards,
                            'n_training_epochs':training_epochs,
                            'n_steps_simulated':steps_simulated,
                            'training_completed':False,
                            }
                if training_filename != None:
                    self.save_dictionary(dictionary=training_results,
                                        filename=training_filename)
            #
            if training_complete:
                # we stop if the stopping criterion was met at the end of
                # the current episode
                training_results['training_completed'] = True
                break
        #
        if not training_complete:
            # if we stopped the training because the maximal number of
            # episodes was reached, we throw a warning
            warning_string = ("Warning: Training was stopped because the "
            "maximum number of episodes, {0}, was reached. But the stopping "
            "criterion has not been met.")
            warnings.warn(warning_string.format(self.N_episodes_max))
        #
        return training_results

    def save_dictionary(self,dictionary,filename):
        """Save a dictionary in hdf5 format"""

        with h5py.File(filename, 'w') as hf:
            for key, value in dictionary.items():
                hf.create_dataset(str(key), 
                    data=value)
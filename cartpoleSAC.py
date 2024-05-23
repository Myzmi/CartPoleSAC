import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from mushroom_rl.environments import Gymnasium
import gymnasium as gym
from mushroom_rl.utils import TorchUtils

from tqdm import trange

import custom_cartpole

import tkinter as tk

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, n_epochs, n_steps, n_steps_test, save, load):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir='./logs' if save else None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    horizon = 200
    gamma = 0.99
    #mdp = Gymnasium('Pendulum-v1', horizon, gamma, headless=False)
    mdp = Gymnasium('CustomCartPole-v0')

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    if load:
        agent = SAC.load('logs/SAC/agent-best.msh')
    else:
        # Approximator
        actor_input_shape = mdp.info.observation_space.shape
        actor_mu_params = dict(network=ActorNetwork,
                               n_features=n_features,
                               input_shape=actor_input_shape,
                               output_shape=mdp.info.action_space.shape)
        actor_sigma_params = dict(network=ActorNetwork,
                                  n_features=n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=mdp.info.action_space.shape)

        actor_optimizer = {'class': optim.Adam,
                           'params': {'lr': 3e-4}}

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)

        critic_params = dict(network=CriticNetwork,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': 3e-4}},
                             loss=F.mse_loss,
                             n_features=n_features,
                             input_shape=critic_input_shape,
                             output_shape=(1,))

        # Agent
        agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)

    J = np.mean(dataset.discounted_return)
    R = np.mean(dataset.undiscounted_return)
    E = agent.policy.entropy(dataset.state)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)

        J = np.mean(dataset.discounted_return)
        R = np.mean(dataset.undiscounted_return)
        E = agent.policy.entropy(dataset.state)

        logger.epoch_info(n+1, J=J, R=R, entropy=E)

        if save:
            logger.log_best_agent(agent, J)

    logger.info('Press a button to visualize cartpole')
    input()
    core.evaluate(n_episodes=1, render=True, record=True)

def startClick():
    noEpoch= int(inputFieldEpoch.get())
    noSteps= int(inputFieldStep.get())

    root.destroy()

    save = False
    load = False
    TorchUtils.set_default_device('cpu')
    experiment(alg=SAC, n_epochs=noEpoch, n_steps=noSteps, n_steps_test=2000, save=save, load=load)

#tkinter page
# Create a window
root = tk.Tk()
root.title("Color Selector")
root.geometry('800x600')

# Configure column to expand horizontally
root.columnconfigure(0, weight=1)

#add form frame
step_frame = tk.LabelFrame(root, padx=20, pady=20, bd=0)
step_frame.grid(row=0, column=0)

# Configure column to expand horizontally
step_frame.columnconfigure(0, weight=1)

#step_frame labels
formLabelCart = tk.Label(step_frame, text="Epochs")
formLabelCart.grid(row=0, column=0, sticky="e")

formLabelStep = tk.Label(step_frame, text="Steps")
formLabelStep.grid(row=1, column=0, sticky="e")

#cart_frame boxes
inputFieldEpoch = tk.Entry(step_frame, width=10)
inputFieldEpoch.insert(0, "20")
inputFieldEpoch.grid(row=0, column=1, sticky="ew")

inputFieldStep = tk.Entry(step_frame, width=10)
inputFieldStep.insert(0, "1000")
inputFieldStep.grid(row=1, column=1, sticky="ew")

#start button
startButton = tk.Button(root, text="Start", command=startClick, width=25, height=3)
startButton.grid(row=2, column=0)

# Start the Tkinter event loop
root.mainloop()
import random

import numpy as np
import torch
from collections import deque

from torch import nn
from networks_QLearning import DQN


class Agents_DQN:
    def __init__(self, obs_space, action_space, nagents: int, nn_type: str = "NN", memory_size: int = 200000,
                 batch_size: int = 32) -> None:
        self.device = "cpu"
        self.memory = deque(maxlen=memory_size)

        self.burnin = 1e4
        self.sync_every = 1e4
        self.current_step = 0
        self.save_every = 5e5

        self.current_exploration_rate = 0
        self.batch_size = batch_size
        self.agents = self.gen_Agents(
            nagents=nagents, obs_space=obs_space, action_space=action_space, nn_type=nn_type)

    def gen_Agents(self, nagents, obs_space, action_space, nn_type) -> np.ndarray:
        agents = np.zeros(nagents, dtype=Agent_DQN)
        for i in range(nagents):
            agents[i] = Agent_DQN(obs_space=obs_space, action_space=action_space,
                                  nn_type=nn_type, device=self.device, id=i, batch_size=self.batch_size,
                                  save_dir="Output")
        return agents

    def store(self, nobs, next_nobs, nactions, nrewards, ndone) -> None:

        nobs = torch.tensor(nobs, device=self.device)
        next_nobs = torch.tensor(next_nobs, device=self.device)
        nactions = torch.tensor(nactions, device=self.device)
        nrewards = torch.tensor(nrewards, device=self.device)
        ndone = torch.tensor(ndone, device=self.device)

        self.current_step += 1
        self.memory.append((nobs, next_nobs, nactions, nrewards, ndone))

    def retrieve(self):
        batch = random.sample(self.memory, self.batch_size)
        nobs, next_nobs, nactions, nrewards, ndone = map(
            torch.stack, zip(*batch))
        return nobs, next_nobs, nactions, nrewards, ndone

    def distributed_learn(self):
        if self.current_step % self.sync_every == 0:
            for agent in self.agents:
                agent.sync_Q_target()
        if self.current_step < self.burnin:
            return None

        nobs, next_nobs, nactions, nrewards, ndone = self.retrieve()
        losses = []
        for idx, agent in enumerate(self.agents):
            losses.append(agent.loss(nobs=nobs.clone()[:, idx, :], next_nobs=next_nobs.clone(
            )[:, idx, :], nactions=nactions.clone()[:, idx], nrewards=nrewards.clone()[:, idx], ndone=ndone[:, idx]))
        for idx, agent in enumerate(self.agents):
            agent.learn(loss=losses[idx])

        return losses[0].item()

    def agents_act(self, nobs):
        nactions = []
        for agent in self.agents:
            action_index, exploration_rate = agent.act(nobs[agent.id])
            nactions.append(action_index)
        self.current_exploration_rate = exploration_rate
        return tuple(nactions)



class Agent_DQN:
    def __init__(self, obs_space, action_space, nn_type, device, id, save_dir, batch_size) -> None:
        self.id = id
        self.lr = 0.001
        self.adam_eps = 0.001
        self.device = device
        self.save_dir = save_dir

        self.counter = 0
        self.sync_every = 1e4

        self.nactions = np.array(action_space)[0].n
        self.gamma = 0.9
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999995
        self.exploration_rate_min = 0.1
        self.batch_size = batch_size

        self.dqn = DQN(input_dim=obs_space,
                       output_dim=action_space, nn_type=nn_type)
        self.target_dqn = DQN(input_dim=obs_space,
                              output_dim=action_space, nn_type=nn_type)

        self.target_dqn.load_state_dict(self.dqn.state_dict())
        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.dqn = self.dqn.to(device=self.device)
        self.target_dqn = self.target_dqn.to(device=self.device)

        self.optimizer = torch.optim.Adam(
            self.dqn.parameters(), lr=self.lr, eps=self.adam_eps)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, obs):
        if np.random.rand() < self.exploration_rate:
            action_index = np.random.randint(self.nactions)
        else:
            obs = torch.tensor(obs, device=self.device)
            action_values = self.dqn.forward(state=obs)
            action_index = int(self.dqn.get_max(x=action_values).item())
        self.update_exploration_rate()

        return action_index, self.exploration_rate

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

    def loss(self, nobs, next_nobs, nactions, nrewards, ndone):
        td_estimate = self.td_estimate(nobs, nactions)
        td_target = self.td_target(nrewards, next_nobs, ndone)
        loss = self.loss_fn(td_estimate, td_target)
        return loss

    def td_estimate(self, state, action):
        action_Qs = self.dqn.forward(state=state)
        current_Q = action_Qs[np.arange(0, self.batch_size), action]
        return current_Q

    def td_target(self, reward, next_state, done):
        next_state_Q = self.dqn.forward(next_state).detach()
        best_action = torch.tensor([int(self.dqn.get_max(nsQ).item()) for nsQ in next_state_Q])
        nextQ = self.target_dqn.forward(next_state).detach()[np.arange(0, self.batch_size), best_action]
        target = (reward + (1 - done.float()) * self.gamma * nextQ).float()
        return target

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), 0.5)
        self.optimizer.step()

    def sync_Q_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())


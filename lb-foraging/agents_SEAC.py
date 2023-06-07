import random

import numpy as np
import torch
from collections import deque

from torch import nn

from Networks import NN_Actor, NN_Critic


class Agents_SEAC:
    def __init__(self, obs_space, action_space, nagents: int, nn_type_actor: str = "NN", nn_type_critic: str = "NN",
                 memory_size: int = 200000, batch_size: int = 32) -> None:
        self.device = "cpu"
        self.memory = deque(maxlen=memory_size)

        self.burnin = 1e4
        self.sync_every = 1e4
        self.current_step = 0
        self.save_every = 5e5

        self.current_exploration_rate = 0
        self.batch_size = batch_size
        self.agents = self.gen_Agents(
            nagents=nagents, obs_space=obs_space, action_space=action_space, nn_type_actor=nn_type_actor,
            nn_type_critic=nn_type_critic)

    def gen_Agents(self, nagents, obs_space, action_space, nn_type_actor, nn_type_critic) -> np.ndarray:
        agents = np.zeros(nagents, dtype=Agent_SEAC)
        for i in range(nagents):
            agents[i] = Agent_SEAC(obs_space=obs_space, action_space=action_space,
                                   nn_type_actor=nn_type_actor, nn_type_critic=nn_type_critic, device=self.device, id=i,
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
        if self.current_step < self.burnin:
            return None, None

        nobs, next_nobs, nactions, nrewards, ndone = self.retrieve()
        actor_losses = []
        critic_losses = []
        for agent in self.agents:
            # if self.current_step % self.save_every == 0:
            #     self.save_agents()
            actor_losses.append(agent.actor_loss(nobs=nobs.clone(), next_nobs=next_nobs.clone(
            ), nactions=nactions.clone(), nrewards=nrewards.clone(), agents=self.agents))
            critic_losses.append(agent.critic_loss(nobs=nobs.clone(), next_nobs=next_nobs.clone(
            ), nactions=nactions.clone(), nrewards=nrewards.clone(), agents=self.agents))
        for idx, agent in enumerate(self.agents):
            agent.learn(
                actor_loss=actor_losses[idx], critic_loss=critic_losses[idx])
            agent.soft_update()

        return actor_losses[0].item(), critic_losses[0].item()

    def agents_act(self, nobs):
        nactions = []
        for agent in self.agents:
            action_index, exploration_rate = agent.act(nobs[agent.id])
            nactions.append(action_index)
        self.current_exploration_rate = exploration_rate
        return tuple(nactions)

    def save_agents(self):
        for agent in self.agents:
            agent.save(self.current_step, self.save_every)


class Agent_SEAC:
    def __init__(self, obs_space, action_space, nn_type_actor, nn_type_critic, device, id, save_dir) -> None:
        self.id = id
        self._gamma = 0.99
        self.lambda_critic = 0.95
        self.lambda_actor = 0.95
        self.lr = 0.001
        self.adam_eps = 0.001
        self.device = device
        self.save_dir = save_dir

        self.counter = 0

        self.nactions = np.array(action_space)[0].n
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999995
        self.exploration_rate_min = 0.1

        self.actor = NN_Actor(input_dim=obs_space,
                              output_dim=action_space, nn_type=nn_type_actor)
        self.actor = self.actor.to(device=self.device)

        self.critic = NN_Critic(input_dim=obs_space, nn_type=nn_type_critic)
        self.target_critic = NN_Critic(input_dim=obs_space, nn_type=nn_type_critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.critic = self.critic.to(device=self.device)
        self.target_critic = self.target_critic.to(device=self.device)

        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, eps=self.adam_eps)
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr, eps=self.adam_eps)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, obs):
        # if self.counter>15000:
        #     print("test")
        obs = torch.tensor(obs, device=self.device)
        action_distribution = self.actor.forward(state=obs)
        if np.random.rand() < self.exploration_rate:
            # action_index = torch.multinomial(action_distribution, 1).item()
            action_index= np.random.randint(self.nactions)
        else:
            action_index = self.actor.get_max(action_distribution)
            action_index = int(action_index.item())
        self.update_exploration_rate()

        # self.counter+=1
        return action_index, self.exploration_rate

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)

    def soft_update(self):
        for eval_param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - 0.005) + eval_param.data * 0.005)

    def other_agents(self, agents):
        if self.id == 0:
            return agents[self.id + 1:]
        elif self.id == len(agents) - 1:
            return agents[:self.id]
        else:
            return np.concatenate((agents[:self.id], agents[self.id + 1:]), axis=0)

    def actor_loss(self, nobs, next_nobs, nactions, nrewards, agents):
        # print("nobs ",nobs)
        individual_policy = torch.gather(self.actor.forward(nobs[:, self.id]), 1, nactions[:, self.id].unsqueeze(1))
        individual_reward = torch.unsqueeze(nrewards[:, self.id], dim=1)
        individual_prediction = self.critic.forward(nobs[:, self.id]).detach()
        individual_estimation = self.target_critic.forward(next_nobs[:, self.id]).detach()  # target
        # if individual_policy.any() < 0.001:
        #     print("test")

        individual_policy = torch.clamp(individual_policy, min=0.01)
        individual_policy_log = -torch.log(individual_policy)
        individual_Bellman = individual_reward + self._gamma * individual_estimation - individual_prediction
        individual_loss = individual_policy_log * individual_Bellman

        # return torch.mean(individual_loss)
        other_loss_list = []
        for other_agent in self.other_agents(agents=agents):
            other_individual_policy = torch.gather(self.actor.forward(nobs[:, other_agent.id]), 1,
                                                   nactions[:, other_agent.id].unsqueeze(1))
            other_policy = torch.gather(other_agent.loss.forward(nobs[:, other_agent.id]), 1,
                                        nactions[:, other_agent.id].unsqueeze(1)).detach()
            other_reward = nrewards[:, other_agent.id]
            other_individual_prediction = self.critic.forward(nobs[:, other_agent.id]).detach()
            other_individual_estimation = self.target_critic.forward(next_nobs[:, other_agent.id]).detach()

            other_individual_policy = torch.clamp(other_individual_policy, min=0.01)
            other_policy = torch.clamp(other_policy, min=0.01)
            policy_scale = other_individual_policy / other_policy
            other_policy_log = torch.log(other_policy)
            other_Bellman = other_reward + self._gamma * other_individual_estimation - other_individual_prediction
            other_loss = policy_scale * other_policy_log * other_Bellman
            other_loss_list.append(other_loss)

        other_loss_final = torch.zeros(other_loss.shape).to(device=self.device)
        for _other_loss in other_loss_list:
            other_loss_final += _other_loss

        other_loss_final = self.lambda_actor * other_loss_final
        final_loss = individual_loss - other_loss_final
        loss = torch.mean(final_loss)
        return loss

    def critic_loss(self, nobs, next_nobs, nactions, nrewards, agents):
        # loss=self.critic_test(nobs, next_nobs, nactions, nrewards, agents)
        individual_reward = torch.unsqueeze(nrewards[:, self.id], dim=1)
        individual_prediction = self.critic.forward(nobs[:, self.id])
        individual_estimation = self.target_critic.forward(next_nobs[:, self.id]).detach()  # target

        y_individual = individual_reward + self._gamma * individual_estimation
        individual_Bellman = individual_prediction - y_individual
        individual_loss = torch.pow(individual_Bellman, 2)

        # return torch.mean(individual_loss)

        other_loss_list = []
        other_agents = self.other_agents(agents=agents)
        for other_agent in other_agents:
            other_individual_policy = torch.gather(self.actor.forward(nobs[:, other_agent.id]), 1,
                                                   nactions[:, other_agent.id].unsqueeze(1)).detach()
            other_policy = torch.gather(other_agent.loss.forward(nobs[:, other_agent.id]), 1,
                                        nactions[:, other_agent.id].unsqueeze(1)).detach()
            other_reward = nrewards[:, other_agent.id]
            other_individual_prediction = self.critic.forward(nobs[:, other_agent.id])
            other_individual_estimation = self.target_critic.forward(next_nobs[:, other_agent.id])

            other_individual_policy = torch.clamp(other_individual_policy, min=0.01)
            other_policy = torch.clamp(other_policy, min=0.01)
            other_policy_scale = other_individual_policy / other_policy
            y_other = other_reward + self._gamma * other_individual_estimation
            other_Bellman = other_individual_prediction - y_other
            other_Bellman_pow = torch.pow(other_Bellman, 2)
            other_loss = other_policy_scale * other_Bellman_pow
            other_loss_list.append(other_loss)

        other_loss_final = torch.zeros(other_loss.shape).to(device=self.device)
        for _other_loss in other_loss_list:
            other_loss_final += _other_loss

        other_loss_final = self.lambda_critic * other_loss_final
        final_loss = individual_loss + other_loss_final
        loss = torch.mean(final_loss)
        return loss

    def learn(self, actor_loss, critic_loss):
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_critic.step()

    def save(self, current_step, save_every):

        save_path = (
            self.save_dir /
            f"agent_{self.id}_{int(current_step // save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.actor.state_dict(),
                 exploration_rate=self.exploration_rate),
            save_path,
        )

        print(f'agent_{self.id} saved to {save_path} at step {current_step}')

    def load(self, path):
        pass

import copy
from tqdm import tqdm
import time
import gym
import numpy as np
import lbforaging
from agents_SEAC import Agents_SEAC


class Train:
    def __init__(self, actor_nn_type: str, critic_nn_type: str, n_agents, n_food, dim, game_count,
                 render: bool = False):
        self.n_agents = n_agents
        self.n_food = n_food
        self.dim = dim

        self.render = render
        self.game_count = game_count

        self.actor_nn_type = actor_nn_type
        self.critic_nn_type = critic_nn_type

    def run(self):

        env = gym.make(f"Foraging-{self.dim}x{self.dim}-{self.n_agents}p-{self.n_food}f-v2")

        seac = Agents_SEAC(obs_space=env.observation_space,
                           action_space=env.action_space, nagents=self.n_agents, nn_type_actor=self.actor_nn_type,
                           nn_type_critic=self.critic_nn_type)

        final_rewards = []
        final_actor_losses = []
        final_critic_losses = []
        for episode in tqdm(range(self.game_count)):
            final_reward, exploration_rate, actor_loss, critic_loss = self.game_loop(env, self.render, seac)
            final_rewards.append(final_reward)
            if actor_loss is not None:
                final_actor_losses.append(actor_loss)
                final_critic_losses.append(critic_loss)
        return final_rewards, final_actor_losses, final_critic_losses

    def game_loop(self, env, render: bool, seac: Agents_SEAC):
        nobs = env.reset()

        done = False

        if render:
            env.render()
            time.sleep(3)

        while not done:
            nactions = seac.agents_act(nobs=nobs)

            next_nobs, nrewards, ndone, _ = env.step(nactions)
            seac.store(nobs=nobs, nactions=nactions, nrewards=nrewards,
                       next_nobs=next_nobs, ndone=ndone)

            nobs = copy.deepcopy(next_nobs)

            if render:
                env.render()
                time.sleep(3)

            done = np.all(ndone)
        actor_loss, critic_loss = seac.distributed_learn()

        return sum(nrewards), seac.current_exploration_rate, actor_loss, critic_loss

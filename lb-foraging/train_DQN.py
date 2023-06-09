import copy
from tqdm import tqdm
import time
import gym
import numpy as np
import lbforaging
from agents_DQN import Agents_DQN


class Train:
    def __init__(self, nn_type: str, n_agents, n_food, dim, game_count,
                 render: bool = False):
        self.n_agents = n_agents
        self.n_food = n_food
        self.dim = dim

        self.render = render
        self.game_count = game_count

        self.nn_type = nn_type

    def run(self):

        env = gym.make(f"Foraging-{self.dim}x{self.dim}-{self.n_agents}p-{self.n_food}f-v2")

        dqn = Agents_DQN(obs_space=env.observation_space,
                         action_space=env.action_space, nagents=self.n_agents, nn_type=self.nn_type)

        final_rewards = []
        losses = []
        for episode in tqdm(range(self.game_count)):
            final_reward, exploration_rate, loss = self.game_loop(env, self.render, dqn)
            final_rewards.append(final_reward)
            if loss is not None:
                losses.append(loss)
        return final_rewards, losses

    def game_loop(self, env, render: bool, dqn: Agents_DQN):
        nobs = env.reset()

        done = False

        if render:
            env.render()
            time.sleep(3)

        while not done:
            nactions = dqn.agents_act(nobs=nobs)

            next_nobs, nrewards, ndone, _ = env.step(nactions)
            dqn.store(nobs=nobs, nactions=nactions, nrewards=nrewards,
                      next_nobs=next_nobs, ndone=ndone)

            nobs = copy.deepcopy(next_nobs)

            if render:
                env.render()
                time.sleep(3)

            done = np.all(ndone)
        loss = dqn.distributed_learn()

        return sum(nrewards), dqn.current_exploration_rate, loss

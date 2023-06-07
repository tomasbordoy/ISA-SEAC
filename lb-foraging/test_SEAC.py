import argparse
import logging
import random
import matplotlib.pyplot as plt
import copy
import torch.autograd
# import DQN-P2-F1-dim12x12_typeNN
from tqdm import tqdm
import time
import gym
import numpy as np
import lbforaging
import sys
# from DQN-P2-F1-dim12x12_typeNN import *
from agents_SEAC import Agents_SEAC
from train_SEAC import Train
import pickle


# torch.autograd.set_detect_anomaly(True)

class Main:
    def __init__(self, runs: int, n_agents: int, n_food: int, dim: int, render: bool, game_count: int, actor_type: str, critic_type: str):
        self.runs = runs
        self.actor_type = actor_type
        self.critic_type = critic_type

        self.n_agents = n_agents
        self.n_food = n_food
        self.dim = dim

        self.title = f'SEAC-P{n_agents}-F{n_food}-dim{dim}x{dim}_type{actor_type}-{critic_type}'
        self.path = f'SEAC-output/{self.title}/'

        self.render = render
        self.game_count = game_count

        self.test()

    def test(self):
        episode_list = [i for i in range(1, self.game_count + 1)]

        test_rewards = []
        test_actor_losses = []
        test_critic_losses = []
        for run in range(self.runs):
            test = Train(actor_nn_type=self.actor_type, critic_nn_type=self.critic_type, n_agents=self.n_agents,
                         n_food=self.n_food, dim=self.dim, game_count=self.game_count, render=self.render)

            rewards, actor_losses, critic_losses = test.run()
            test_rewards.append(rewards)
            test_actor_losses.append(actor_losses)
            test_critic_losses.append(critic_losses)

            del test

        data = (episode_list, test_rewards, test_actor_losses, test_critic_losses)
        with open(self.path+self.title+".pkl", "wb") as f:
            pickle.dump(data, f)












"""

logger = logging.getLogger(__name__)


# rp1 = ReplayBuffer(max_size=1000, input_shape=env.observ)


def _game_loop(env, render: bool, seac: Agents_SEAC):

    nobs = env.reset()

    done = False

    if render:
        env.render()
        time.sleep(3)

    while not done:
        nactions = seac.agents_act(nobs=nobs)
        # for i in range(10):
        #     print(env.action_space.sample())
        next_nobs, nrewards, ndone, _ = env.step(nactions)
        seac.store(nobs=nobs, nactions=nactions, nrewards=nrewards,
                   next_nobs=next_nobs, ndone=ndone)
        # print(nobs)
        # print("next",next_nobs,"\n")
        nobs = copy.deepcopy(next_nobs)

        # if sum(nrewards) > 0:
        #     print(nrewards)
        if render:
            env.render()
            time.sleep(3)
            # print(env.field)
            # print("---------------")
        done = np.all(ndone)
    actor_loss, critic_loss = seac.distributed_learn()

    return sum(nrewards), seac.current_exploration_rate, actor_loss, critic_loss


def learning_curve(episodes, rewards, exploration_rate):
    fig, ax = plt.subplots()
    avg_every = max(int(episodes[len(rewards[0]) - 1] / 100), 1)

    # Convert rewards to a numpy array
    rewards = np.array(rewards)

    # Calculate mean and standard deviation of x and y values
    mean_y = []
    std_y = []
    for i in range(0, rewards.shape[1], avg_every):
        mean_y.append(np.mean(rewards[:, i:i + avg_every], axis=None))
        std_y.append(np.std(rewards[:, i:i + avg_every], axis=None))

    # Plot mean line
    ax.plot(range(1, len(mean_y) + 1), mean_y, color='blue', label='Mean')

    # Shade region around mean line representing x and y errors
    if len(rewards) > 1:
        ax.fill_between(range(1, len(mean_y) + 1), np.array(mean_y) - np.array(std_y),
                        np.array(mean_y) + np.array(std_y), color='blue', alpha=0.5)

    # Add legend and labels
    ax.legend()
    ax.set_xlabel(f'Episodes (averaged every {avg_every} episodes)')
    ax.set_ylabel('Rewards')
    ax.set_title(
        f'Learning Curve - number of episodes: {episodes[len(rewards[0]) - 1]}\n Exploration_Rate: {exploration_rate:.3f}')

    # Show plot
    plt.savefig(f"learning_curve_ep{episodes[len(rewards[0]) - 1]}.png")
    # plt.show()


def loss_curve(losses, episodes, title, file_name):
    fig, ax = plt.subplots()
    print("losses len ", len(losses))
    print("pre_len ", len(episodes))
    episodes = episodes[-len(losses):]

    print("after_len ", len(episodes))
    print("last_len ", episodes[-1])
    ax.plot(episodes, losses)
    # ax.set_xlim(left=episodes[0], right=episodes[-1])
    ax.legend()
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    plt.savefig(file_name)


def main(game_count: int = 1, render: bool = False, runs: int = 1):
    run_rewards = []
    episode_list = [i for i in range(1, game_count + 1)]
    for run in range(runs):
        n_agents = 2
        n_food = 1
        dim = 12
        env = gym.make(f"Foraging-{dim}x{dim}-{n_agents}p-{n_food}f-v2")
        # env = ForagingEnv(players=n_agents, max_player_level=2, field_size=(
        #     12, 12), max_food=1, sight=2, max_episode_steps=25, force_coop=False)
        seac = Agents_SEAC(obs_space=env.observation_space,
                           action_space=env.action_space, nagents=n_agents)
        # nobs = env.reset()

        final_rewards = []
        final_actor_losses = []
        final_critic_losses = []
        for episode in tqdm(range(game_count)):
            if episode % 5000 == 0 and runs == 1 and episode != 0:
                learning_curve(episodes=episode_list, rewards=[final_rewards], exploration_rate=exploration_rate)
            final_reward, exploration_rate, actor_loss, critic_loss = _game_loop(env, render, seac)
            final_rewards.append(final_reward)
            if actor_loss is not None:
                final_actor_losses.append(actor_loss)
                final_critic_losses.append(critic_loss)
        run_rewards.append(final_rewards)
    print("final_actor_losses: ", final_actor_losses)
    print("final_critic_losses: ", final_critic_losses)
    print("length of final_actor_losses: ", len(final_actor_losses))
    print("length of final_critic_losses: ", len(final_critic_losses))
    learning_curve(episodes=episode_list, rewards=run_rewards, exploration_rate=exploration_rate)
    loss_curve(losses=final_actor_losses, episodes=episode_list, title="actor",
               file_name="expl_actor_losses_actor-95_critic-95.png")
    loss_curve(losses=final_critic_losses, episodes=episode_list, title="critic",
               file_name="expl_critic_losses_actor-95_critic-95.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=10000, help="How many times to run the game"
    )

    args = parser.parse_args()
    args.render = False
    main(args.times, args.render, 2)
"""
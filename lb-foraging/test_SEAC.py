from train_SEAC import Train
import pickle


class Main:
    def __init__(self, runs: int, n_agents: int, n_food: int, dim: int, render: bool, game_count: int, actor_type: str,
                 critic_type: str):
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
        with open(self.path + self.title + ".pkl", "wb") as f:
            pickle.dump(data, f)

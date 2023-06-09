from train_DQN import Train
import pickle


class Main:
    def __init__(self, runs: int, n_agents: int, n_food: int, dim: int, render: bool, game_count: int, type: str):
        self.runs = runs
        self.type = type

        self.n_agents = n_agents
        self.n_food = n_food
        self.dim = dim

        self.title = f'DQN-P{n_agents}-F{n_food}-dim{dim}x{dim}_type{type}'
        self.path = f'DQN-output/{self.title}/'

        self.render = render
        self.game_count = game_count

        self.test()

    def test(self):
        episode_list = [i for i in range(1, self.game_count + 1)]

        test_rewards = []
        test_losses = []
        for run in range(self.runs):
            test = Train(nn_type=self.type, n_agents=self.n_agents,
                         n_food=self.n_food, dim=self.dim, game_count=self.game_count, render=self.render)

            rewards, loss = test.run()
            test_rewards.append(rewards)
            test_losses.append(loss)

            del test

        data = (episode_list, test_rewards, test_losses)
        with open(self.path+self.title+".pkl", "wb") as f:
            pickle.dump(data, f)

import matplotlib.pyplot as plt
import pickle
from scipy.stats import sem


class Plot:

    def plot(self, rewards_list, episode_list, name, color, k, title, ylabel) -> None:
        rewards_list = [[sum(sub_list[i:i + k]) / k for i in range(0, len(sub_list), k)] for sub_list in rewards_list]
        episode_list = episode_list[::k]
        ziped_rewards_list = list(zip(*rewards_list))
        column_average = [sum(sub_list) / len(sub_list) for sub_list in ziped_rewards_list]
        column_sem = [sem(sub_list) for sub_list in ziped_rewards_list]
        x = episode_list[-len(column_average):]
        y = column_average
        errorp = [i1 + i2 for i1, i2 in zip(y, column_sem)]
        errorm = [i1 - i2 for i1, i2 in zip(y, column_sem)]
        plt.plot(x, y, color=color, label=name)
        if title == "Actor loss curve":
            plt.ylim(-1,1)
        elif title==  "Critic loss curve":
            plt.ylim(0,1)

        plt.title(title)
        plt.xlabel("Episodes trained")
        plt.ylabel(ylabel)
        plt.legend(loc="upper right")
        plt.fill_between(x, errorm, errorp, alpha=0.2, edgecolor=color, facecolor=color)

    def save_plot(self, name):
        plt.show()
        plt.savefig(name, dpi=300)

types = ["NN","BNN"]

full_plot=True
path_list=[]
name_list=[]
colors = ['r', 'b']

titles=["Learning curve","Loss curve"]
ylabels=["Reward","Loss"]
names = ["Learning_curve.png","Loss_curve.png"]

# titles=["Learning curve"]
# ylabels=["Reward"]
# names = ["Learning_curve.png"]


for type in types:

    path = f"DQN-output/DQN-P2-F1-dim12x12_type{type}/DQN-P2-F1-dim12x12_type{type}.pkl"
    path_list.append(path)
    name_list.append(f"{type}")

if full_plot:
    episodes=[]
    rewards=[]
    losses=[]
    for i, path in enumerate(path_list):
        with open(path, "rb") as f:
            episode, reward, loss = pickle.load(f)
            episodes.append(episode)
            rewards.append(reward)
            losses.append(loss)
    data=[rewards, losses]
    plot = Plot()


    for j,title in enumerate(titles):
        plt.figure()
        for i,name in enumerate(name_list):
            plot.plot(rewards_list=data[j][i], episode_list=episodes[i], name=name, color=colors[i], title=title,
                      ylabel=ylabels[j], k=int(len(episodes[i]) / 10000 * 50))
        plot.save_plot(name=names[j])

else:
    with open(path_list[0], "rb") as f:
        episode, reward, loss = pickle.load(f)
    plt.figure()
    test = Plot()
    test.plot(rewards_list=reward, episode_list=episode, name="NN-NN", color=colors[0],title="Learning curve", ylabel="Reward",k=50)
    test.save_plot(name="learningcurve.png")
    plt.figure()
    test = Plot()
    test.plot(rewards_list=loss, episode_list=episode, name="NN-NN", color=colors[1], title="Loss curve", ylabel="Loss", k=50)
    test.save_plot(name="loss.png")
    plt.figure()


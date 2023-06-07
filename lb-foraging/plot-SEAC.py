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

actor_types = ["NN", "BNN"]
critic_types = ["NN", "BNN"]

full_plot=True
path_list=[]
name_list=[]
colors = ['g', 'b', 'r', 'c']

titles=["Learning curve","Actor loss curve","Critic loss curve"]
ylabels=["Reward","Loss","Loss"]
names = ["Learning_curve.png","Actor_loss_curve.png","Critic_loss_curve.png"]

full_plot=True
path_list=[]
name_list=[]
colors = ['g', 'b', 'r', 'c']


for actor_type in actor_types:
    for critic_type in critic_types:
        path = f"SEAC-output/SEAC-P2-F1-dim12x12_type{actor_type}-{critic_type}/P2-F1-dim12x12_type{actor_type}-{critic_type}.pkl"
        path_list.append(path)
        name_list.append(f"{actor_type}-{critic_type}")

if full_plot:
    episodes=[]
    rewards=[]
    actors=[]
    critics=[]
    for i, path in enumerate(path_list):
        with open(path, "rb") as f:
            episode, reward, actor, critic = pickle.load(f)
            episodes.append(episode)
            rewards.append(reward)
            actors.append(actor)
            critics.append(critic)
    data=[rewards,actors,critics]
    plot = Plot()


    for j,title in enumerate(titles):
        plt.figure()
        for i,name in enumerate(name_list):
            plot.plot(rewards_list=data[j][i], episode_list=episodes[i], name=name, color=colors[i], title=title,
                      ylabel=ylabels[j], k=int(len(episodes[i]) / 10000 * 50))
        plot.save_plot(name=names[j])

else:
    with open(path_list[0], "rb") as f:
        episode, reward, actor, critic = pickle.load(f)
    plt.figure()
    test = Plot()
    test.plot(rewareds_list=reward, episode_list=episode, name="NN-NN", color=colors[0],title="Learning curve", ylabel="Reward",k=50)
    test.save_plot(name="learningcurve.png")
    plt.figure()
    test = Plot()
    test.plot(rewards_list=actor, episode_list=episode, name="NN-NN", color=colors[1],title="Actor loss curve", ylabel="Loss",k=50)
    test.save_plot(name="actorloss.png")
    plt.figure()
    test = Plot()
    test.plot(rewards_list=critic, episode_list=episode, name="NN-NN", color=colors[2],title="Critic loss curve", ylabel="Loss",k=50)
    test.save_plot(name="criticloss.png")


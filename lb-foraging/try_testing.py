import torch
import pickle

with open("DQN-output/DQN-P2-F1-dim12x12_typeNN/DQN-P2-F1-dim12x12_typeNN.pkl", "rb") as f:
    episode, reward, actor = pickle.load(f)

print(actor)



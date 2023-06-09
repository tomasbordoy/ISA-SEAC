# Multi-Agent Reinforcement Learning on Social Dilemma Problems

Over the past few years, the field of Reinforcement Learning has managed to take innovative steps toward the development of successful applications using single-agent strategies. However, many real-world scenarios involve multiple agents operating in the same environment, each pursuing its own rewards. The emergence of social dilemmas in multi-agent environments occurs when agents act greedily, prioritizing their own rewards without considering the common good. Such self-centered behaviors can result in conflicts, resource depletion, or inefficient utilization of the environment's resources. Consequently, it becomes imperative for the agents to adapt their strategies and behaviors to obtain general benefits, fostering collaboration and coordination among themselves.  

In this paper, we focus on the use of multi-agent reinforcement learning (MARL) for social dilemma problems using the Level Based Foraging (LBF) environment, where agents must balance the competing goals of maximizing their individual rewards while also learning to cooperate and coordinate with other agents. We will also be using probabilistic approaches by using Bayesian Neural Networks (BNN), which may be useful in MARL due to their ability to model uncertainty and capture complex dependencies in the environment.

## Installation

Clone the repository and add the virtual environment called "venv7"

## Usage

Depending on the algorithm desired, run the respective file "run-SEAC" or "run-DQN". After the execution is completed, pickle files will have been generated in the output folders. After that run the according plot scripts to visualize the learning curve and the loss functions.

## Contributing

**Authors:**

- Tomás Bordoy García-Carpintero

- Lucas Alexander Damberg Torp Dyssel

**Supervisors:**

- Melih Kandemir

- Oliver Baumann



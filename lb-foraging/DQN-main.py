from test_DQN import Main


types = ["NN","BNN"]
runs = 3
n_agents = 2
n_food = 1
dim = 12
render = False
game_count = 100000
full_test=True

if full_test:
    for type in types:
        main = Main(runs=runs, n_agents=n_agents, n_food=n_food, dim=dim, render=render, game_count=game_count,
                    type=type)
else:
    main = Main(runs=runs, n_agents=n_agents, n_food=n_food, dim=dim, render=render, game_count=game_count,
                type=types[0])
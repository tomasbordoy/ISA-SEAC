from test_SEAC import Main

actor_types = ["NN","BNN"]
critic_types = ["NN","BNN"]
runs = 3
n_agents = 2
n_food = 1
dim = 12
render = False
game_count = 100000
full_test=True

if full_test:
    for actor_type in actor_types:
        for critic_type in critic_types:
            main = Main(runs=runs, n_agents=n_agents, n_food=n_food, dim=dim, render=render, game_count=game_count,
                        actor_type=actor_type,
                        critic_type=critic_type)
else:
    main = Main(runs=runs, n_agents=n_agents, n_food=n_food, dim=dim, render=render, game_count=game_count,
                actor_type=actor_types[0],
                critic_type=critic_types[0])
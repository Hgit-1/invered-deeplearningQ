initial_epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 10000

final_epsilon = initial_epsilon * (epsilon_decay ** num_episodes)
print(f"Final epsilon after {num_episodes} episodes: {final_epsilon}")

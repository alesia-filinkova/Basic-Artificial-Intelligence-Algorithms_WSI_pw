import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Q_training:
    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=0.1, num_episodes=5000):
        self.env = gym.make("CliffWalking-v0")

        # Hiperparametres Q-learning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Parametr eksploracji
        self.num_episodes = num_episodes
    
    def train(self):
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()  # Eksploracja
                else:
                    action = np.argmax(self.q_table[state])  # Eksploatacja
                # print(self.env.step(action))
                next_state, reward, done, _, _ = self.env.step(action)
                # TODO q-table actualisation 
                delta = reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action]
                self.q_table[state, action] += self.learning_rate * delta
                state = next_state
    
    def terminal_visualization(self):
        self.env.reset()
        arrows = np.array(['↑', '→', '↓', '←'])
        rows, columns = self.env.unwrapped.shape
        # print(self.env.unwrapped.shape)
        field = np.full((rows, columns), ' ')
        best_way = np.argmax(self.q_table, axis=1)
        ways = best_way.reshape((rows, columns))
        cliff_positions = np.where(self.env.unwrapped._cliff == 1)

        for row in range(rows):
            for col in range(columns):
                state = row * columns + col
                if (row, col) in zip(cliff_positions[0], cliff_positions[1]):
                    field[row, col] = 'C'
                elif (row, col) == (rows - 1, columns - 1): 
                    field[row, col] = 'E'
                else:
                    field[row, col] = arrows[ways[row, col]]
        for r in field:
            print(' '.join(r))

    def qtable_directions_map(self, qtable, map_size):
        """Get the best learned action and map it to arrows."""
        qtable_val_max = qtable.max(axis=1).reshape(map_size)
        qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size)
        directions = {0: "↑", 1: "→", 2: "↓", 3: "←" }
        qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
        eps = np.finfo(float).eps  # Minimum float number on the machine
        for idx, val in enumerate(qtable_best_action.flatten()):
            if qtable_val_max.flatten()[idx] > eps:
                qtable_directions[idx] = directions[val]
        qtable_directions = qtable_directions.reshape(map_size)
        return qtable_val_max, qtable_directions
    
    def plot_q_values_map(self):
        """Plot the learned Q-values and the best actions on the grid."""
        map_size = self.env.unwrapped.shape
        qtable_val_max, qtable_directions = self.qtable_directions_map(self.q_table, map_size)

        # Plot the policy
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            qtable_val_max,
            annot=qtable_directions,
            fmt="",
            ax=ax,
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.5,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "large"},
        ).set(title="Learned Q-values\nArrows represent best action")
        plt.show()
    
    def visualization(self):
        self.terminal_visualization()
        self.plot_q_values_map()
        self.env.close()


if __name__ == "__main__":
    q_training = Q_training()
    q_training.train()
    q_training.visualization()


# def plot_states_actions_distribution(states, actions, map_size):
#     """Plot the distributions of states and actions."""
#     labels = {"LEFT": 3, "DOWN": 2, "RIGHT": 1, "UP": 0}

#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
#     sns.histplot(data=states, ax=ax[0], kde=True)
#     ax[0].set_title("States Distribution")
#     sns.histplot(data=actions, ax=ax[1])
#     ax[1].set_xticks(list(labels.values()), labels=labels.keys())
#     ax[1].set_title("Actions Distribution")
#     fig.tight_layout()
#     plt.show()

# # Setup environment and parameters
# rewards = np.random.rand(len(episodes), 1)  # Placeholder for rewards
# steps = np.random.randint(1, 100, size=(len(episodes), 1))  # Placeholder for steps
# states = np.random.randint(0, env.observation_space.n, size=1000)  # Placeholder states
# actions = np.random.randint(0, env.action_space.n, size=1000)  # Placeholder actions

# # Visualize results
# plot_states_actions_distribution(states, actions, map_size)
# plot_q_values_map(q_table, env, map_size)

# env.close()
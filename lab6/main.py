import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


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
    
    def visualization(self):
        self.terminal_visualization()
        self.env.close()


if __name__ == "__main__":
    q_training = Q_training()
    q_training.train()
    q_training.visualization()

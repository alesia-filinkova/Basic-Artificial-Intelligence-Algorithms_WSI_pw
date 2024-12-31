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
        # print(self.env.reset())
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
                best_next_action = np.argmax(self.q_table[next_state])
                delta = reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action]
                self.q_table[state, action] += self.learning_rate * delta
                state = next_state


if __name__ == "__main__":
    q_training = Q_training()
    q_training.train()

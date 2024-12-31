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

        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        # print(self.env.reset())
    
    def train(self):
        for episode in range(self.num_episodes):
            start, _ = self.env.reset()
            done = False


if __name__ == "__main__":
    q_training = Q_training()
    q_training.train()

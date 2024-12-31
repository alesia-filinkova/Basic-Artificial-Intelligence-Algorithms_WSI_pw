import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")

    # Hiperparametres Q-learning
    learning_rate = 0.1 
    discount_factor = 0.99 
    epsilon = 0.1  # Parametr eksploracji
    num_episodes = 500

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

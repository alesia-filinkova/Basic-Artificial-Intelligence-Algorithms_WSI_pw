import numpy as np


class EvolutionAlgorithm:
    def __init__(self, cities_matrix, population_size=5, generations=100, mutation_rate=0.01):
        self.cities_matrix = cities_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.create_population()

    def create_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [0] + list(np.random.permutation(np.arange(1, len(self.cities_matrix) - 1))) + [len(self.cities_matrix) - 1]
            population.append(individual)
        return population

import numpy as np
from solution_utils import evaluate_solution, validate_solution


class EvolutionAlgorithm:
    def __init__(
        self, cities_matrix, population_size=350, generations=100, mutation_rate=0.01, crossover_rate=0.9
    ):
        self.cities_matrix = cities_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.create_population()

    def create_population(self):
        population = []
        for _ in range(self.population_size):
            individual = (
                [0]
                + list(np.random.permutation(np.arange(1, len(self.cities_matrix) - 1)))
                + [len(self.cities_matrix) - 1]
            )
            population.append(individual)
        return population

    def evaluate_population(self):
        fitness_scores = []
        for individual in self.population:
            distance = evaluate_solution(self.cities_matrix, individual)
            fitness_scores.append(1 / distance)
        return fitness_scores

    def select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        return [self.population[i] for i in parents_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            start, end = sorted(np.random.choice(range(1, len(parent1) - 1), size=2, replace=False))
            child = parent1[:start] + [None] * (end - start) + parent1[end:]
            pointer = 0
            for gene in parent2:
                if gene not in child:
                    while child[pointer] is not None:
                        pointer += 1
                    child[pointer] = gene
            return child
        return parent1

    def mutate(self, child):
        for i in range(1, len(child) - 1):
            if np.random.rand() < self.mutation_rate:
                j = np.random.randint(1, len(child) - 1)
                child[i], child[j] = child[j], child[i]

    def run(self):
        for _ in range(self.generations):
            fitness_scores = self.evaluate_population()
            parents = self.select_parents(fitness_scores)
            next_generation = []
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.extend([child1, child2])
            self.population = next_generation
        evaluations = [evaluate_solution(self.cities_matrix, i) for i in self.population]
        best_index = evaluations.index(min(evaluations))
        best_solution = self.population[best_index]
        validate_solution(self.cities_matrix, best_solution)
        return best_solution

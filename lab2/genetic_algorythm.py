import numpy as np
from solution_utils import evaluate_solution, validate_solution


class EvolutionAlgorithm:
    def __init__(
        self,
        cities_matrix,
        population_size=100,
        generations=100,
        mutation_rate=0.05,
        crossover_rate=0.9,
        tournament_size=5,
        # number of the best individuals that are guaranteed to pass into the next generation without changes
        elitism_size=2, 
    ):
        self.cities_matrix = cities_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.best_distances_per_generation = []
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
        return [evaluate_solution(self.cities_matrix, ind) for ind in self.population]

    def tournament_selection(self, distances):
        """select parents through a tournament"""
        selected = []
        for _ in range(self.population_size):
            contenders_idx = np.random.choice(range(self.population_size), self.tournament_size)
            best_idx = min(contenders_idx, key=lambda i: distances[i])
            selected.append(self.population[best_idx])
        return selected

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            start, end = sorted(
                np.random.choice(range(1, len(parent1) - 1), size=2, replace=False)
            )
            child = [None] * len(parent1)
            child[start:end] = parent1[start:end]

            pointer = 1
            for gene in parent2[1:-1]:
                if gene not in child:
                    while child[pointer] is not None:
                        pointer += 1
                    child[pointer] = gene

            child[0], child[-1] = 0, len(parent1) - 1
            return child
        return parent1.copy()

    def mutate(self, child):
        if np.random.rand() < self.mutation_rate:
            i, j = sorted(np.random.choice(range(1, len(child) - 1), size=2, replace=False))
            child[i:j] = reversed(child[i:j])

    def run(self):
        for _ in range(self.generations):
            distances = self.evaluate_population()

            best_idx = int(np.argmin(distances))
            best_distance = distances[best_idx]
            self.best_distances_per_generation.append(best_distance)

            sorted_indices = np.argsort(distances)
            elites = [self.population[i] for i in sorted_indices[:self.elitism_size]]
            
            parents = self.tournament_selection(distances)

            next_generation = elites.copy()
            while len(next_generation) < self.population_size:
                idx = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = parents[idx[0]], parents[idx[1]]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.extend([child1, child2])

            self.population = next_generation[:self.population_size]

        # final check
        final_distances = self.evaluate_population()
        best_index = int(np.argmin(final_distances))
        best_solution = self.population[best_index]
        validate_solution(self.cities_matrix, best_solution)
        return best_solution

import cupy as cp # type: ignore
import random
from individual import Individual

############################## SELECCIÓN #########################################################

class TournamentSelection:
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size

    def select(self, population):
        """Selecciona el individuo con mejor fitness dentro de un subconjunto aleatorio."""
        selected = random.sample(population.individuals, self.tournament_size)
        return min(selected, key=lambda ind: ind.fitness).clone()

############################## CROSSOVER #########################################################

class OnePointCrossover:
    def __init__(self):
        self.rng = cp.random.default_rng()

    def apply(self, parent1, parent2):
        """Realiza un cruce de un solo punto entre dos padres"""
        # Asegurar que ambos padres tengan la misma longitud
        min_length = min(len(parent1), len(parent2))

        # Punto de cruce aleatorio (evitar los extremos)
        point = int(self.rng.integers(1, min_length))

        # Crear hijos intercambiando segmentos
        child1_genes = cp.concatenate((parent1.genes[:point], parent2.genes[point:min_length]))
        child2_genes = cp.concatenate((parent2.genes[:point], parent1.genes[point:min_length]))

        # Si los padres tenían longitudes diferentes, completar con el padre más largo
        if len(parent1) > min_length:
            child1_genes = cp.concatenate((child1_genes, parent1.genes[min_length:]))
        if len(parent2) > min_length:
            child2_genes = cp.concatenate((child2_genes, parent2.genes[min_length:]))

        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)

        return child1, child2

############################## MUTACIÓN #########################################################

class GaussianMutation:
    def __init__(self):
        self.rng = cp.random.default_rng()
    
    def mutate(self, individual, mutation_rate, param_bounds):
        """Aplica una mutación aleatoria a los parámetros con una tasa dada"""
        # Crea una copia del individuo para no modificar el original
        mutated = individual.clone()

        # Aplicar mutación a cada parámetro
        for i in range(len(mutated)):
            if self.rng.random() < mutation_rate:
                if i < len(param_bounds):
                    low, high = param_bounds[i]

                    # Mutación gaussiana con límites
                    mutation_strength = 0.1 * (high - low)
                    mutation = self.rng.normal(0, mutation_strength)
                    mutated.genes[i] = mutated.genes[i] + mutation

                    # Aplicar límites
                    mutated.genes[i] = cp.clip(mutated.genes[i], low, high)
                else:
                    # Si no hay límites definidos, mutación pequeña
                    mutated.genes[i] += self.rng.uniform(-0.01, 0.01)

        return mutated
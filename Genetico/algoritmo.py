import cupy as cp  # type: ignore
import os
from operadores_geneticos import TournamentSelection, OnePointCrossover, GaussianMutation
from fitness import FitnessEvaluator
from population import Population
from lectura_datos import leer_incendio_referencia

class GeneticAlgorithm:
    def __init__(self, tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, ctx,
                 archivo_preentrenado=None, generacion_preentrenada=0, num_steps=10000, batch_size=10, num_combustibles = 5,
                 ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None, ajustar_ignicion=True,
                 ignicion_fija_x=None, ignicion_fija_y=None, verbose=False):
        
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.limite_parametros = limite_parametros
        self.ruta_incendio_referencia = ruta_incendio_referencia
        self.ctx = ctx
        self.archivo_preentrenado = archivo_preentrenado
        self.generacion_preentrenada = generacion_preentrenada
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_combustibles = num_combustibles
        self.ajustar_beta_gamma=ajustar_beta_gamma
        self.beta_fijo=beta_fijo
        self.gamma_fijo=gamma_fijo
        self.ajustar_ignicion=ajustar_ignicion
        self.ignicion_fija_x=ignicion_fija_x
        self.ignicion_fija_y=ignicion_fija_y
        self.verbose = verbose

        # Operadores
        self.selection_op = TournamentSelection()
        self.crossover_op = OnePointCrossover()
        self.mutation_op = GaussianMutation()

        self.evaluator = FitnessEvaluator(self.ctx)

        self.incendio_referencia = leer_incendio_referencia(self.ruta_incendio_referencia)
        self.celdas_quemadas_referencia = cp.where(self.incendio_referencia > 0.001, 1, 0)
    
    def initialize(self):
        if self.archivo_preentrenado: # Si hay una población preentrenada la carga
            poblacion = Population.cargar_poblacion_preentrenada(
                self.archivo_preentrenado, 
                self.tamano_poblacion, 
                self.limite_parametros,
                num_combustibles=self.num_combustibles,
                ajustar_beta_gamma=self.ajustar_beta_gamma, 
                ajustar_ignicion=self.ajustar_ignicion,
                verbose=self.verbose
            )
        else: # Instancio la población inicial
            poblacion = Population.initial_population(self.tamano_poblacion, self.limite_parametros)
            poblacion = self.evaluate_population(poblacion)
        
        return poblacion
    
    def _validate_and_extract_params(self, parametros_batch):
        evaluator = self.evaluator
        params = []

        if self.ajustar_beta_gamma and self.ajustar_ignicion: # Exp2
            for D, A, B, x, y, betas, gammas in parametros_batch:
                A, B = evaluator.validate_courant_and_adjust(A, B)
                x, y = evaluator.validate_ignition_point(x, y, self.incendio_referencia, self.limite_parametros)
                betas, gammas = evaluator.validate_beta_gamma(betas, gammas)
                params.append((D, A, B, x, y, betas, gammas))
        elif self.ajustar_beta_gamma and not self.ajustar_ignicion: # Exp3
            for genes in parametros_batch:
                D, A, B = genes[0], genes[1], genes[2]
                betas = genes[3:8]
                gammas = genes[8:13]
                A, B = evaluator.validate_courant_and_adjust(A, B)
                betas, gammas = evaluator.validate_beta_gamma(betas, gammas)
                params.append((D, A, B, betas, gammas))
        else: # Exp1
            for D, A, B, x, y in parametros_batch:
                A, B = evaluator.validate_courant_and_adjust(A, B)
                x, y = evaluator.validate_ignition_point(x, y, self.incendio_referencia, self.limite_parametros)
                params.append((D, A, B, x, y))
        
        return params
    
    def evaluate_population(self, poblacion):
        """Procesa una población en batches para aprovechar el paralelismo"""  
        population_fitness = []

        for i in range(0, len(poblacion), self.batch_size):
            # Accede a los individuos del batch
            batch = poblacion.individuals[i:i+self.batch_size]
            if self.verbose:
                print(f'Procesando batch {i//self.batch_size + 1} de {len(poblacion) // self.batch_size}...')
            # Accede a los parámetros de los individuos del batch
            parametros_batch = [individuo.genes for individuo in batch]

            # Validación de parámetros
            parametros_validados = self._validate_and_extract_params(parametros_batch)

            # Lista de valores de fitness 
            fitness_values = self.evaluator.evaluate_batch(parametros_validados, self.celdas_quemadas_referencia, self.num_steps, 
                                       ajustar_beta_gamma=self.ajustar_beta_gamma, beta_fijo=self.beta_fijo, gamma_fijo=self.gamma_fijo, 
                                       ajustar_ignicion=self.ajustar_ignicion, ignicion_fija_x=self.ignicion_fija_x, 
                                       ignicion_fija_y=self.ignicion_fija_y)

            population_fitness.extend(fitness_values)

        for individuo, fitness in zip(poblacion.individuals, population_fitness):
            individuo.update_fitness(fitness)

        return poblacion
    
    def produce_offspring(self, poblacion, mutation_rate):
        new_population = []
        for _ in range(self.tamano_poblacion // 2):
            parent1 = self.selection_op.select(poblacion)
            parent2 = self.selection_op.select(poblacion)
            child1, child2 = self.crossover_op.apply(parent1, parent2)
            child1 = self.mutation_op.mutate(child1, mutation_rate, self.limite_parametros)
            new_population.extend([child1, child2])
        return Population(new_population, generation=poblacion.generation + 1)
    
    def apply_elitism(self, poblacion, elite):
        peor = poblacion.worst()
        elite_clonada = elite.clone()
        peor_idx = poblacion.individuals.index(peor)
        poblacion.individuals[peor_idx] = elite_clonada
        return poblacion
    
    def step(self, poblacion, mutation_rate, gen):
        elite = poblacion.best()
        nueva = self.produce_offspring(poblacion, mutation_rate)
        nueva = self.evaluate_population(nueva)
        nueva = self.apply_elitism(nueva, elite)
        return nueva
    
    def run(self):
        resultados_dir = "resultados"
        os.makedirs(resultados_dir, exist_ok=True)

        poblacion = self.initialize()
        mutation_rate = 0.3 * 0.99**self.generacion_preentrenada

        for gen in range(self.generaciones + 1):
            if gen > 0 and self.verbose:
                print(f"Iniciando generación {gen}...")
                poblacion = self.step(poblacion, mutation_rate, gen)
                mutation_rate *= 0.99
            
            if self.verbose:
                print(f"Generación {gen}: Mejor fitness = {poblacion.best().fitness}")

            if self.verbose:
                for i, ind in enumerate(poblacion.individuals, 1):
                    print(f"Individuo {i}: {ind.as_dict(self.ajustar_beta_gamma, self.ajustar_ignicion)}")
            
            poblacion.guardar_resultados(resultados_dir, gen - 1 + self.generacion_preentrenada,
                                    num_combustibles=self.num_combustibles,
                                    ajustar_beta_gamma=self.ajustar_beta_gamma, 
                                    ajustar_ignicion=self.ajustar_ignicion,
                                    verbose=self.verbose)
            
        if self.verbose:
            print(f"Resultados guardados en: {resultados_dir}")
        return poblacion

############################## ALGORITMO GENÉTICO #########################################################

def genetic_algorithm(tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, ctx,
                 archivo_preentrenado=None, generacion_preentrenada=0, num_steps=10000, batch_size=10, num_combustibles=5,
                 ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None, ajustar_ignicion=True,
                 ignicion_fija_x=None, ignicion_fija_y=None, verbose=False):
    
    ga = GeneticAlgorithm(tamano_poblacion, generaciones, limite_parametros, ruta_incendio_referencia, ctx,
                 archivo_preentrenado, generacion_preentrenada, num_steps, batch_size, num_combustibles,
                 ajustar_beta_gamma, beta_fijo, gamma_fijo, ajustar_ignicion,
                 ignicion_fija_x, ignicion_fija_y, verbose)
    
    return ga.run()
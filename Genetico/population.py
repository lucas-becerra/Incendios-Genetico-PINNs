import cupy as cp
from individual import Individual
import os, csv
import numpy as np

class Population:
    def __init__(self, individuals, generation=0):
        self.individuals = individuals
        self.generation = generation
    
    @classmethod
    def initial_population(cls, size, limite_parametros):
        """
        Genera población inicial dentro de los límites establecidos
        Args:
            size: Número de individuos
            limite_parametros: Lista de tuplas (min, max) para cada parámetro
        Returns:
            Lista de individuos (cada individuo es una instancia de Individual)
        """
        rng = cp.random.default_rng()
        n_params = len(limite_parametros)
    
        # Genera números aleatorios uniformes en [0,1]
        rand = rng.random((size, n_params), dtype=cp.float32)
    
        # Convierte límites a arrays
        lows  = cp.array([low for low, _ in limite_parametros], dtype=cp.float32)
        highs = cp.array([high for _, high in limite_parametros], dtype=cp.float32)

        # Escala cada columna al rango correspondiente
        genes = lows + rand * (highs - lows)
        individuals = [Individual(genes[i]) for i in range(size)]

        return cls(individuals, generation=0)
    
    def __len__(self):
        return len(self.individuals)
    
    def best(self):
        return min([ind for ind in self.individuals if ind.fitness is not None], key=lambda ind: ind.fitness)
    
    def worst(self):
        return max([ind for ind in self.individuals if ind.fitness is not None], key=lambda ind: ind.fitness)
    
    def sort_by_fitness(self):
        self.individuals.sort(key=lambda ind: ind.fitness)
        
    def to_array(self):
        return cp.asarray([ind.genes for ind in self.individuals])
    
    def replace_worst(self, individual):
        self.sort_by_fitness()
        self.individuals[-1] = individual 

    def mean_fitness(self):
        fitness_values = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if not fitness_values:
            return None
        return sum(fitness_values) / len(fitness_values)

    @classmethod
    def from_results(cls, resultados, ajustar_beta_gamma, ajustar_ignicion):
        individuals = []
        for res in resultados:
            D, A, B = res['D'], res['A'], res['B']
            fitness = res['fitness']

            if ajustar_beta_gamma and ajustar_ignicion: # Exp2
                x, y = res['x'], res['y']
                betas, gammas = res['betas'], res['gammas']
                genes = cp.array([D, A, B, x, y, betas, gammas])
            
            elif ajustar_beta_gamma and not ajustar_ignicion: # Exp3
                betas, gammas = res['betas'], res['gammas']
                genes = cp.concatenate([cp.array([D, A, B]), betas, gammas])
        
            else: # Exp1
                x, y = res['x'], res['y']
                genes = cp.array([D, A, B, x, y])
        
            individuals.append(Individual(genes, fitness))

        return cls(individuals)
    
    @classmethod
    def cargar_poblacion_preentrenada(cls, archivo_preentrenado, tamano_poblacion, limite_parametros, 
                                    num_combustibles=5, ajustar_beta_gamma=True, ajustar_ignicion=True, verbose=False):
        """Carga una población preentrenada desde un CSV"""

        if verbose:
            print(f"[DEBUG] Cargando población preentrenada desde: {archivo_preentrenado}")

        if not os.path.exists(archivo_preentrenado):
            raise ValueError(f"[DEBUG] Archivo {archivo_preentrenado} no encontrado.")

        poblacion_cargada = []
        with open(archivo_preentrenado, 'r') as f:
            reader = csv.DictReader(f)
            total_rows = sum(1 for _ in open(archivo_preentrenado)) - 1  # sin header
            f.seek(0)  # volver al inicio después del conteo

            for idx, row in enumerate(reader, start=1):
                try:
                    D = float(row['D']); A = float(row['A']); B = float(row['B'])

                    if ajustar_ignicion:
                        x = int(float(row['x'])); y = int(float(row['y']))

                    if ajustar_beta_gamma and 'beta_1' in row: # Exp3
                        betas = cp.array([float(row[f'beta_{i}']) for i in range(1, num_combustibles+1)],
                                        dtype=cp.float32)
                        gammas = cp.array([float(row[f'gamma_{i}']) for i in range(1, num_combustibles+1)],
                                        dtype=cp.float32)
                    elif ajustar_beta_gamma and 'beta' in row: # Exp2
                        betas = cp.array(float(row['beta']), dtype=cp.float32)
                        gammas = cp.array(float(row['gamma']), dtype=cp.float32)

                    fval = row.get('fitness', '')
                    fitness = (float(fval) if (fval is not None and fval != '') else None)

                    if ajustar_beta_gamma and ajustar_ignicion:  # Exp2
                        poblacion_cargada.append({
                            "D": D, "A": A, "B": B, "x": x, "y": y,
                            "betas": betas, "gammas": gammas, "fitness": fitness
                        })
                        # Prints de debug solo en casos selectos
                        if verbose and (idx <= 3 or idx == total_rows // 2 or idx == total_rows):
                            print(f"[DEBUG] Fila {idx}/{total_rows}: "
                                f"D={D}, A={A}, B={B}, x={x}, y={y}, "
                                f"betas={betas.get()}, gammas={gammas.get()}, fitness={fitness}")
                    elif ajustar_beta_gamma and not ajustar_ignicion:    # Exp3
                        poblacion_cargada.append({
                            "D": D, "A": A, "B": B,
                            "betas": betas, "gammas": gammas, "fitness": fitness
                        })
                        # Prints de debug solo en casos selectos
                        if verbose and (idx <= 3 or idx == total_rows // 2 or idx == total_rows):
                            print(f"[DEBUG] Fila {idx}/{total_rows}: "
                                f"D={D}, A={A}, B={B}, "
                                f"betas={betas.get()}, gammas={gammas.get()}, fitness={fitness}")
                    else:                                        # Exp1
                        poblacion_cargada.append({
                            "D": D, "A": A, "B": B, "x": x, "y": y, "fitness": fitness 
                        })
                        # Prints de debug solo en casos selectos
                        if verbose and (idx <= 3 or idx == total_rows // 2 or idx == total_rows):
                            print(f"[DEBUG] Fila {idx}/{total_rows}: "
                                f"D={D}, A={A}, B={B}, x={x}, y={y}, fitness={fitness}")

                except (ValueError, KeyError) as e:
                    if verbose and idx <= 5:  # solo aviso explícito en las primeras filas
                        print(f"[WARNING] Fila inválida {idx}, se salta. Error: {e}")
                    continue

        # Ajustar tamaño de la población
        num_cargados = len(poblacion_cargada)
        if verbose:
            print(f"\n[DEBUG] Total individuos cargados: {num_cargados}")

        if num_cargados == 0:
            raise ValueError("[DEBUG] No se cargaron individuos válidos.")

        if num_cargados > tamano_poblacion:
            if verbose:
                print(f"[DEBUG] Se cargaron {num_cargados}, recortando a {tamano_poblacion}")
            rng = cp.random.default_rng()
            indices = rng.choice(num_cargados, tamano_poblacion, replace=False)
            poblacion_cargada = [poblacion_cargada[i] for i in indices.get()]
        elif num_cargados < tamano_poblacion:
            faltantes = tamano_poblacion - num_cargados
            if verbose:
                print(f"[DEBUG] Faltan {faltantes} individuos. Generando población inicial para completar.")
            nuevos_pop = cls.initial_population(faltantes, limite_parametros)

            # Reformatear cada nuevo individuo con las mismas claves que los cargados
            for ind in nuevos_pop.individuals:
                ind_dict = ind.as_dict(ajustar_beta_gamma, ajustar_ignicion)
                poblacion_cargada.append(ind_dict)

        if verbose:
            print(f"[DEBUG] Población final: {len(poblacion_cargada)} individuos.")
        return cls.from_results(poblacion_cargada, ajustar_beta_gamma, ajustar_ignicion)

    def guardar_resultados(self, resultados_dir, gen, num_combustibles=5, 
                       ajustar_beta_gamma=True, ajustar_ignicion=True,
                       verbose=False):
        """Guarda resultados en un archivo CSV"""
    
        csv_filename = os.path.join(resultados_dir, f'resultados_generacion_{gen+1}.csv')
    
        # Definir nombres de columnas dinámicamente
        if ajustar_beta_gamma and ajustar_ignicion:   # Exp2
            fieldnames = ['D', 'A', 'B', 'x', 'y'] \
                        + ['beta'] \
                        + ['gamma'] \
                        + ['fitness']
        elif ajustar_beta_gamma and not ajustar_ignicion:  # Exp3
            fieldnames = ['D', 'A', 'B'] \
                        + [f'beta_{i}' for i in range(1, num_combustibles+1)] \
                        + [f'gamma_{i}' for i in range(1, num_combustibles+1)] \
                        + ['fitness']
        else:                                                      # Exp1
            fieldnames = ['D', 'A', 'B', 'x', 'y'] + ['fitness'] 

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
            for individuo in self.individuals:
                resultado = individuo.as_dict(ajustar_beta_gamma, ajustar_ignicion)
                row = {
                    'D': resultado['D'],
                    'A': resultado['A'],
                    'B': resultado['B'],
                    'fitness': resultado['fitness'],
                }

                if ajustar_ignicion:
                    row['x'] = resultado['x']
                    row['y'] = resultado['y']
            
                if ajustar_beta_gamma:
                    betas = resultado['betas']
                    gammas = resultado['gammas']

                    # Convertir a array siempre
                    if not isinstance(betas, (cp.ndarray, np.ndarray)):
                        betas = cp.array([betas], dtype=cp.float32)
                    if not isinstance(gammas, (cp.ndarray, np.ndarray)):
                        gammas = cp.array([gammas], dtype=cp.float32)

                    if betas.size > 1:
                        # Expandir betas
                        for i, beta in enumerate(betas, start=1):
                            row[f'beta_{i}'] = float(beta)  # aseguro que sea serializable
                        # Expandir gammas
                        for i, gamma in enumerate(gammas, start=1):
                            row[f'gamma_{i}'] = float(gamma)
                    else:
                        row['beta'] = float(betas)
                        row['gamma'] = float(gammas)
                writer.writerow(row)
    
        if verbose:
            print(f"✅ Resultados guardados en {csv_filename}")
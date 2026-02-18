import cupy as cp

class Individual:
    def __init__(self, genes, fitness=None):
        self.genes = genes
        self.fitness = fitness

    def clone(self):
        genes_copy = cp.copy(self.genes) if hasattr(cp, "array") else self.genes[:]
        return Individual(genes_copy, self.fitness)

    def invalidate_fitness(self):
        self.fitness = None
    
    def update_fitness(self, new_fitness):
        self.fitness = new_fitness
    
    def __len__(self):
        return len(self.genes)
    
    def to_array(self):
        return cp.asarray(self.genes)

    def as_dict(self, ajustar_beta_gamma=True, ajustar_ignicion=True, num_combustibles=5):
        """Convierte el individuo a diccionario para guardar en CSV.
        
        Args:
            ajustar_beta_gamma: Si se ajustan parámetros beta/gamma
            ajustar_ignicion: Si se ajusta punto de ignición
            num_combustibles: Número de tipos de combustibles (para Exp3)
        
        Returns:
            Diccionario con los parámetros del individuo
        """
        # Mapea genes
        D, A, B = self.genes[:3]

        result = {"D": float(D), "A": float(A), "B": float(B), "fitness": self.fitness}

        if ajustar_beta_gamma and ajustar_ignicion: # Exp2
            x, y = self.genes[3], self.genes[4]
            betas = self.genes[5]
            gammas = self.genes[6]
            result.update({"x": int(x), "y": int(y), "betas": betas, "gammas": gammas})

        elif ajustar_beta_gamma and not ajustar_ignicion: # Exp3
            # Usar num_combustibles para extraer la cantidad correcta de parámetros
            betas = self.genes[3:3+num_combustibles]
            gammas = self.genes[3+num_combustibles:3+2*num_combustibles]
            result.update({"betas": betas, "gammas": gammas})
        
        else: # Exp1
            x, y = self.genes[3], self.genes[4]
            result.update({"x": int(x), "y": int(y)})

        return result

    
import cupy as cp # type: ignore
from config import d, dt
import cupyx.scipy.ndimage # type: ignore
import sys
import os

# Agrega el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo_rdc import spread_infection_adi, courant

class FitnessEvaluator:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rng = cp.random.default_rng()

    def validate_courant_and_adjust(self, A, B):
        """Valida la condición de Courant y ajusta parámetros si es necesario."""
        
        iteraciones = 0
        while not courant(dt/2, A, B, d, self.ctx.wx, self.ctx.wy, h_dx=self.ctx.h_dx, h_dy=self.ctx.h_dy):
            iteraciones += 1
            # Alternativa más eficiente: seleccionar aleatoriamente entre 0, 1
            param_idx = int(self.rng.integers(0, 2))  # 0, 1

            if param_idx == 0:  # A
                A = cp.array(A * float(self.rng.uniform(0.8, 0.99)), dtype=cp.float32)
            elif param_idx == 1:  # B
                B = cp.array(B * float(self.rng.uniform(0.8, 0.99)), dtype=cp.float32)

        # Evitar bucles infinitos
            if iteraciones > 100:
                print(f"Warning: Validación Courant tomó {iteraciones} iteraciones")
                break
        return A, B

    def validate_ignition_point(self, x, y, incendio_referencia, limite_parametros):
        """Valida que el punto de ignición tenga combustible o que esté en el incendio de referencia."""
        lim_x, lim_y = limite_parametros[3], limite_parametros[4]
        while self.ctx.vegetacion[int(y), int(x)] <= 2 or incendio_referencia[int(y), int(x)] <= 0.001:
            x, y = float(self.rng.integers(lim_x[0], lim_x[1])), float(self.rng.integers(lim_y[0], lim_y[1]))
        return x, y

    def validate_beta_gamma(self, betas, gammas):
        """Valida los parámetros beta y gamma. Beta[i] > Gamma[i] para todo i"""
        mask = gammas >= betas
        gammas[mask] = 0.9 * betas[mask]
        return betas, gammas
    
    def _apply_veg_mapping(self, vegetacion, beta_params, gamma_params, veg_types):
        beta_map = cp.zeros_like(vegetacion, dtype=cp.float32)
        gamma_map = cp.zeros_like(vegetacion, dtype=cp.float32)

        for j, veg_type in enumerate(veg_types):
            mask = (vegetacion == int(veg_type))
            if j < len(beta_params) and j < len(gamma_params):
                beta_map = cp.where(mask, beta_params[j], beta_map)
                gamma_map = cp.where(mask, gamma_params[j], gamma_map)

        return beta_map, gamma_map
    
    def _create_fuel_map(self, parametros_batch, ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None):
        """Crea mapas de beta y gamma personalizados para cada simulación en el batch
        basados en los parámetros optimizados y el tipo de vegetación."""
        batch_size = len(parametros_batch)
        vegetacion = self.ctx.vegetacion
        ny, nx = self.ctx.ny, self.ctx.nx
    
        # Crear arrays de salida
        beta_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
        gamma_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
        # Obtener tipos únicos de vegetación
        veg_types = cp.array([3, 4, 5, 6, 7], dtype=cp.int32)
    
        # Para cada simulación en el batch
        for i, params in enumerate(parametros_batch):
            # Inicializar con valores por defecto
            beta_map = cp.zeros_like(vegetacion, dtype=cp.float32)
            gamma_map = cp.zeros_like(vegetacion, dtype=cp.float32)
            if ajustar_beta_gamma and len(params) == 7: # Exp2
                beta_map = cp.full_like(vegetacion, params[5], dtype=cp.float32)
                gamma_map = cp.full_like(vegetacion, params[6], dtype=cp.float32)
            else:
                if ajustar_beta_gamma and len(params) == 5: # Exp3
                    beta_params, gamma_params = params[3], params[4]
                else:   # Exp1
                    beta_params, gamma_params = beta_fijo, gamma_fijo
            
                beta_map, gamma_map = self._apply_veg_mapping(vegetacion, beta_params, gamma_params, veg_types)
        
            beta_batch[i] = beta_map
            gamma_batch[i] = gamma_map

        sigma = (0, 10.0, 10.0)

        # Suavizado de los mapas
        beta_batch = cupyx.scipy.ndimage.gaussian_filter(beta_batch, sigma=sigma)
        gamma_batch = cupyx.scipy.ndimage.gaussian_filter(gamma_batch, sigma=sigma)

        return beta_batch, gamma_batch
    
    def evaluate_batch(self, parametros_batch, burnt_cells, num_steps=10000, ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None,
                  ajustar_ignicion=True, ignicion_fija_x=None, ignicion_fija_y=None):
        """Calcula el fitness para múltiples combinaciones de parámetros en paralelo."""
    
        batch_size = len(parametros_batch)
        vegetacion = self.ctx.vegetacion
        wx = self.ctx.wx
        wy = self.ctx.wy
        h_dx_mapa = self.ctx.h_dx
        h_dy_mapa = self.ctx.h_dy
        ny, nx = self.ctx.ny, self.ctx.nx

        print(f'Batch size: {batch_size}')
    
        # Inicializar arrays para el batch
        S_batch = cp.ones((batch_size, ny, nx), dtype=cp.float32)
        I_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
        R_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
        # Configurar puntos de ignición para cada simulación
        for i, params in enumerate(parametros_batch):
            # Extraer coordenadas (D, A, B, x, y, ...)
            if ajustar_ignicion:
                x, y = int(params[3]), int(params[4])
            else: 
                x, y = ignicion_fija_x, ignicion_fija_y
            
            S_batch[i, y, x] = 0
            I_batch[i, y, x] = 1

        # Arrays para los nuevos estados
        S_new_batch = cp.empty_like(S_batch)
        I_new_batch = cp.empty_like(I_batch)
        R_new_batch = cp.empty_like(R_batch)
    
        # Crear mapas de parámetros de vegetación personalizados
        beta_batch, gamma_batch = self._create_fuel_map(parametros_batch, ajustar_beta_gamma=ajustar_beta_gamma,
                                                           beta_fijo=beta_fijo, gamma_fijo=gamma_fijo)

        # Crear arrays de parámetros D, A, B para cada simulación
        D_batch = cp.array([param[0] for param in parametros_batch], dtype=cp.float32)
        A_batch = cp.array([param[1] for param in parametros_batch], dtype=cp.float32)
        B_batch = cp.array([param[2] for param in parametros_batch], dtype=cp.float32)

        # Simular en paralelo
        simulaciones_validas = cp.ones(batch_size, dtype=cp.bool_)

        print(f'Numero de pasos a simular: {num_steps}')
        paso_explosion = cp.full(batch_size, -1, dtype=cp.int32)  # -1 significa no explotó
    
        for t in range(num_steps):
            # Llamar al kernel con todos los parámetros necesarios
            spread_infection_adi(
                S=S_batch, I=I_batch, R=R_batch, 
                S_new=S_new_batch, I_new=I_new_batch, R_new=R_new_batch,
                dt=dt, d=d, beta=beta_batch, gamma=gamma_batch,
                D=D_batch, wx=wx, wy=wy, 
                h_dx=h_dx_mapa, h_dy=h_dy_mapa, A=A_batch, B=B_batch, vegetacion=vegetacion
            )

            # Intercambiar arrays
            S_batch, S_new_batch = S_new_batch, S_batch
            I_batch, I_new_batch = I_new_batch, I_batch
            R_batch, R_new_batch = R_new_batch, R_batch
        
            # Verificar si alguna simulación explota
            # Condiciones más estrictas para detectar problemas temprano
            validas = cp.all((R_batch >= -1e-6) & (R_batch <= 1 + 1e-6) & 
                            (I_batch >= -1e-6) & (I_batch <= 1 + 1e-6) &
                            (S_batch >= -1e-6) & (S_batch <= 1 + 1e-6), axis=(1, 2))
        
            # Detectar valores extremos que pueden causar problemas
            valores_extremos = cp.any((R_batch > 10) | (R_batch < -10), axis=(1, 2))
        
            # Registrar el paso de explosión para simulaciones que acaban de explotar
            nuevas_explosiones = simulaciones_validas & (~validas | valores_extremos)
            paso_explosion = cp.where(nuevas_explosiones, t + 1, paso_explosion)
        
            # Actualizar estado de validez
            simulaciones_validas &= validas & (~valores_extremos)
        
            # Si todas las simulaciones explotaron, terminar
            if not cp.any(simulaciones_validas):
                print(f"Todas las simulaciones explotaron en el paso {t+1}")
                break

        # Calcular fitness para cada simulación en paralelo
        fitness_values = []
    
        # Crear máscaras para celdas quemadas (todo el batch de una vez)
        burnt_cells_sim_batch = cp.where(R_batch > 0.001, 1, 0)  # Shape: (batch_size, ny, nx)
    
        # Expandir burnt_cells para el batch
        burnt_cells_expanded = cp.broadcast_to(burnt_cells[cp.newaxis, :, :], (batch_size, ny, nx))
    
        # Calcular unión e intersección para todo el batch
        union_batch = cp.sum(burnt_cells_expanded | burnt_cells_sim_batch, axis=(1, 2))  # Shape: (batch_size,)
        interseccion_batch = cp.sum(burnt_cells_expanded & burnt_cells_sim_batch, axis=(1, 2))  # Shape: (batch_size,)
    
        # Calcular fitness para todo el batch
        burnt_cells_total = cp.sum(burnt_cells)
        fitness_batch = (union_batch - interseccion_batch) / burnt_cells_total

        # Procesar resultados
        for i in range(batch_size):
            fitness_values.append(float(fitness_batch[i]))

        return fitness_values
import cupy as cp # type: ignore
from config import d, dt
import cupyx.scipy.ndimage # type: ignore
import sys
import os

# Agrega el directorio padre al path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modelo_rdc import spread_infection_adi, courant

class FitnessEvaluator:
    def __init__(self, ctx, veg_types=None, verbose=False,
                 nan_check_interval=10, rescue_max_retries=3, gamma_reduction_factor=0.9):
        """
        Inicializa el evaluador de fitness.
        
        Args:
            ctx: Contexto con datos del incendio (vegetación, viento, pendiente, etc.)
            veg_types: Lista de tipos de vegetación combustible a considerar.
                      Si es None (por defecto), detecta automáticamente todos los tipos
                      presentes en ctx.vegetacion (excluyendo valores <= 2 que son no-combustibles).
                      Si se especifica manualmente, debe coincidir con num_combustibles en Exp3.
                      Ejemplo: [3, 4, 5, 6, 7] para todos los tipos, o [3, 5, 6, 7] si falta el tipo 4.
            verbose: Si True, imprime mensajes de debug
        """
        self.ctx = ctx
        self.rng = cp.random.default_rng()
        self.verbose = verbose
        self.nan_check_interval = nan_check_interval
        self.rescue_max_retries = rescue_max_retries
        self.gamma_reduction_factor = gamma_reduction_factor

        # Detectar tipos de vegetación únicos si no se especifican
        if veg_types is None:
            unique_veg = cp.unique(ctx.vegetacion)
            # Filtrar valores <= 2 (no combustibles: 0=sin datos, 1=agua, 2=urbano/rocas)
            self.veg_types = cp.sort(unique_veg[unique_veg > 2]).astype(cp.int32)
        else:
            self.veg_types = cp.array(veg_types, dtype=cp.int32)

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
                if self.verbose:
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
    
    def _reduce_max_gamma(self, gamma_map, factor):
        """Reduce todos los gammas multiplicando por factor (más efectivo para prevenir NaN)"""
        return gamma_map * factor
    
    def _apply_veg_mapping(self, vegetacion, beta_params, gamma_params, veg_types):
        beta_map = cp.zeros_like(vegetacion, dtype=cp.float32)
        gamma_map = cp.zeros_like(vegetacion, dtype=cp.float32)

        for j, veg_type in enumerate(veg_types):
            mask = (vegetacion == int(veg_type))
            if j < len(beta_params) and j < len(gamma_params):
                beta_map = cp.where(mask, beta_params[j], beta_map)
                gamma_map = cp.where(mask, gamma_params[j], gamma_map)

        return beta_map, gamma_map
    
    def _create_fuel_map(self, parametros_batch, ajustar_beta_gamma=True, beta_fijo=None, gamma_fijo=None, ajustar_ignicion=True):
        """Crea mapas de beta y gamma personalizados para cada simulación en el batch
        basados en los parámetros optimizados y el tipo de vegetación.
        
        Args:
            parametros_batch: Lista de parámetros para cada simulación
                - Exp1 (not ajustar_beta_gamma): (D, A, B, x, y)
                - Exp2 (ajustar_beta_gamma, ajustar_ignicion): (D, A, B, x, y, beta, gamma)
                - Exp3 (ajustar_beta_gamma, not ajustar_ignicion): (D, A, B, betas_array, gammas_array)
            ajustar_beta_gamma: Si True, usa beta/gamma de params; si False, usa beta_fijo/gamma_fijo
            beta_fijo: Array de betas fijos para Exp1
            gamma_fijo: Array de gammas fijos para Exp1
            ajustar_ignicion: Si True, indica Exp2 (beta/gamma escalares); si False, Exp3 (arrays)
        """
        batch_size = len(parametros_batch)
        vegetacion = self.ctx.vegetacion
        ny, nx = self.ctx.ny, self.ctx.nx

        # Crear arrays de salida
        beta_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
        gamma_batch = cp.zeros((batch_size, ny, nx), dtype=cp.float32)
    
        # Usar los tipos de vegetación configurados en __init__
        veg_types = self.veg_types
    
        # Para cada simulación en el batch
        for i, params in enumerate(parametros_batch):
            # Inicializar con valores por defecto
            beta_map = cp.zeros_like(vegetacion, dtype=cp.float32)
            gamma_map = cp.zeros_like(vegetacion, dtype=cp.float32)
            
            if not ajustar_beta_gamma:
                # Exp1: Usa valores fijos
                beta_params, gamma_params = beta_fijo, gamma_fijo
                beta_map, gamma_map = self._apply_veg_mapping(vegetacion, beta_params, gamma_params, veg_types)
            elif ajustar_ignicion:
                # Exp2: Beta/gamma son escalares (params = D, A, B, x, y, beta, gamma)
                beta_map = cp.full_like(vegetacion, params[5], dtype=cp.float32)
                gamma_map = cp.full_like(vegetacion, params[6], dtype=cp.float32)
            else:
                # Exp3: Beta/gamma son arrays (params = D, A, B, betas_array, gammas_array)
                beta_params, gamma_params = params[3], params[4]
                beta_map, gamma_map = self._apply_veg_mapping(vegetacion, beta_params, gamma_params, veg_types)
        
            beta_batch[i] = beta_map
            gamma_batch[i] = gamma_map

        sigma = (0, 10.0, 10.0)

        # Suavizado de los mapas
        beta_batch = cupyx.scipy.ndimage.gaussian_filter(beta_batch, sigma=sigma)
        gamma_batch = cupyx.scipy.ndimage.gaussian_filter(gamma_batch, sigma=sigma)

        return beta_batch, gamma_batch
    
    def _simulate_single_with_maps(self, beta_map, gamma_map, x, y, num_steps, D, A, B):
        vegetacion = self.ctx.vegetacion
        wx, wy = self.ctx.wx, self.ctx.wy
        h_dx_mapa, h_dy_mapa = self.ctx.h_dx, self.ctx.h_dy
        ny, nx = self.ctx.ny, self.ctx.nx

        S = cp.ones((1, ny, nx), dtype=cp.float32)
        I = cp.zeros_like(S)
        R = cp.zeros_like(S)
        S[0, y, x] = 0
        I[0, y, x] = 1

        S_new = cp.empty_like(S)
        I_new = cp.empty_like(S)
        R_new = cp.empty_like(S)

        beta_b = beta_map[cp.newaxis, :, :]
        gamma_b = gamma_map[cp.newaxis, :, :]
        D_b = cp.array([D], dtype=cp.float32)
        A_b = cp.array([A], dtype=cp.float32)
        B_b = cp.array([B], dtype=cp.float32)

        for t in range(num_steps):
            spread_infection_adi(
                S=S, I=I, R=R, S_new=S_new, I_new=I_new, R_new=R_new, 
                dt=dt, d=d, beta=beta_b, gamma=gamma_b, D=D_b,
                wx=wx, wy=wy, h_dx=h_dx_mapa, h_dy=h_dy_mapa, 
                A=A_b, B=B_b, vegetacion=vegetacion
            )

            S, S_new = S_new, S
            I, I_new = I_new, I
            R, R_new = R_new, R

            if (t + 1) % self.nan_check_interval == 0 or (t + 1) == num_steps:
                if not bool(cp.all(cp.isfinite(S) & cp.isfinite(I) & cp.isfinite(R)).item()):
                    return None # falló por NaN/Inf
        
        return R[0]
    
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

        if self.verbose:
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
                                                           beta_fijo=beta_fijo, gamma_fijo=gamma_fijo,
                                                           ajustar_ignicion=ajustar_ignicion)

        # Crear arrays de parámetros D, A, B para cada simulación
        D_batch = cp.array([param[0] for param in parametros_batch], dtype=cp.float32)
        A_batch = cp.array([param[1] for param in parametros_batch], dtype=cp.float32)
        B_batch = cp.array([param[2] for param in parametros_batch], dtype=cp.float32)

        # Simular en paralelo
        simulaciones_validas = cp.ones(batch_size, dtype=cp.bool_)

        if self.verbose:
            print(f'Numero de pasos a simular: {num_steps}')
    
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
        
            # Las simulaciones son válidas si no tiene valores NaN
            if (t + 1) % self.nan_check_interval == 0 or (t + 1) == num_steps:
                finite_ok = cp.all(cp.isfinite(S_batch) & cp.isfinite(I_batch) & cp.isfinite(R_batch), axis=(1,2))
                simulaciones_validas &= finite_ok

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

        # Identificamos las simulaciones que fallaron
        failed_idx = cp.where(~simulaciones_validas)[0]
        
        if self.verbose and len(failed_idx) > 0:
            print(f"[RESCUE] Detectadas {len(failed_idx)} simulaciones fallidas con NaN/Inf")
            print(f"[RESCUE] Índices: {failed_idx.tolist()}")

        for i in failed_idx.tolist():
            if ajustar_ignicion:
                x, y = int(parametros_batch[i][3]), int(parametros_batch[i][4])
            else:
                x, y = ignicion_fija_x, ignicion_fija_y

            if self.verbose:
                print(f"[RESCUE] Simulación {i}: Intentando rescatar (D={parametros_batch[i][0]:.4f}, A={parametros_batch[i][1]:.6f}, B={parametros_batch[i][2]:.4f})")

            gamma_try = gamma_batch[i].copy()
            rescued = False

            for retry_num in range(self.rescue_max_retries):
                gamma_try = self._reduce_max_gamma(gamma_try, self.gamma_reduction_factor)
                
                if self.verbose:
                    gamma_max_val = float(cp.max(gamma_try))
                    print(f"  [RESCUE] Intento {retry_num + 1}/{self.rescue_max_retries}: gamma_max = {gamma_max_val:.6f}")
                
                R_ok = self._simulate_single_with_maps(
                    beta_map=beta_batch[i], gamma_map=gamma_try, x=x, y=y, num_steps=num_steps,
                    D=parametros_batch[i][0], A=parametros_batch[i][1], B=parametros_batch[i][2]
                )

                if R_ok is not None:
                    burnt_sim = (R_ok > 0.001)
                    union = cp.sum(burnt_cells | burnt_sim)
                    inter = cp.sum(burnt_cells & burnt_sim)
                    fitness_values[i] = float((union - inter) / cp.sum(burnt_cells))
                    rescued = True
                    
                    if self.verbose:
                        print(f"  [RESCUE] Éxito en intento {retry_num + 1}! Fitness = {fitness_values[i]:.6f}")
                    break
            
            if not rescued:
                fitness_values[i] = float("inf")
                if self.verbose:
                    print(f"[RESCUE] Simulación {i}: Fracaso después de {self.rescue_max_retries} intentos. Fitness = inf")
        
        # Estadísticas finales de rescate
        if self.verbose and len(failed_idx) > 0:
            rescatadas = sum(1 for i in failed_idx.tolist() if fitness_values[i] != float("inf"))
            no_rescatadas = len(failed_idx) - rescatadas
            print(f"[RESCUE] Resumen: {rescatadas}/{len(failed_idx)} simulaciones rescatadas, {no_rescatadas} sin resolver")

        return fitness_values
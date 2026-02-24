from modelo_rdc import spread_infection_adi, courant_batch
import numpy as np 
import cupy as cp # type: ignore
import cupyx.scipy.ndimage #type: ignore
from mapas.io_mapas import preprocesar_datos
import argparse

def get_default_beta_gamma(exp):
    if exp == 1 or exp == 3:
        beta_default = [0.91, 0.72, 1.38, 1.94, 0.75]
        gamma_default = [0.5, 0.38, 0.84, 0.45, 0.14]
    elif exp == 2:
        beta_default = [1.5, 1.5, 1.5, 1.5, 1.5]
        gamma_default = [0.5, 0.5, 0.5, 0.5, 0.5]
    else:
        raise ValueError(f"Experimento {exp} no está definido")

    return beta_default, gamma_default


def get_default_ignition(exp):
    if exp == 1 or exp == 2:
        return [400], [600]
    if exp == 3:
        return [1130, 1300, 620], [290, 150, 280]
    raise ValueError(f"Experimento {exp} no está definido")


parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=1, help="Número de experimento")
parser.add_argument("--num_steps", type=int, default=500, help="Número de pasos de simulación")
parser.add_argument("--d", type=float, default=30.0, help="Tamaño de celda en metros")
parser.add_argument("--dt", type=float, default=0.5, help="Paso temporal en horas")
parser.add_argument("--D", type=float, default=10.0, help="Coeficiente de difusión")
parser.add_argument("--A", type=float, default=1e-4, help="Constante adimensional de viento")
parser.add_argument("--B", type=float, default=15.0, help="Constante de pendiente")
parser.add_argument("--beta", type=float, nargs='+', default=None, help="Lista de beta por tipo de vegetación (tipos 3,4,5,6,7)")
parser.add_argument("--gamma", type=float, nargs='+', default=None, help="Lista de gamma por tipo de vegetación (tipos 3,4,5,6,7)")
parser.add_argument("--ignition_x", type=int, nargs='+', default=None, help="Coordenadas x del/los punto(s) de ignición")
parser.add_argument("--ignition_y", type=int, nargs='+', default=None, help="Coordenadas y del/los punto(s) de ignición")
parser.add_argument("--check_interval", type=int, default=10, help="Cada cuántos pasos validar estabilidad numérica")
parser.add_argument("--state_min", type=float, default=-1e3, help="Valor mínimo permitido en S, I, R para test de estabilidad")
parser.add_argument("--state_max", type=float, default=1e3, help="Valor máximo permitido en S, I, R para test de estabilidad")
parser.add_argument("--visualizar_mapas", action='store_true', help="Visualizar los mapas al final de la simulación")
args = parser.parse_args()

exp = args.exp
num_steps = args.num_steps
visualizar_mapas = args.visualizar_mapas

if num_steps <= 0:
    parser.error("--num_steps debe ser mayor a 0")

default_beta, default_gamma = get_default_beta_gamma(exp)
beta_params = args.beta if args.beta is not None else default_beta
gamma_params = args.gamma if args.gamma is not None else default_gamma

if len(beta_params) != len(gamma_params):
    parser.error("--beta y --gamma deben tener la misma cantidad de parámetros")

if len(beta_params) != 5:
    parser.error("--beta y --gamma deben tener 5 valores (para tipos de vegetación 3,4,5,6,7)")

if args.ignition_x is None and args.ignition_y is None:
    ignition_x, ignition_y = get_default_ignition(exp)
elif args.ignition_x is None or args.ignition_y is None:
    parser.error("Debe proporcionar ambos argumentos: --ignition_x y --ignition_y")
else:
    ignition_x = args.ignition_x
    ignition_y = args.ignition_y

if len(ignition_x) != len(ignition_y):
    parser.error("--ignition_x y --ignition_y deben tener la misma cantidad de coordenadas")

if len(ignition_x) == 0:
    parser.error("Debe indicar al menos un punto de ignición")

if args.d <= 0:
    parser.error("--d debe ser mayor a 0")

if args.dt <= 0:
    parser.error("--dt debe ser mayor a 0")

if args.D < 0:
    parser.error("--D debe ser mayor o igual a 0")

if args.A < 0 or args.B < 0:
    parser.error("--A y --B deben ser mayores o iguales a 0")

if any(not np.isfinite(value) for value in beta_params + gamma_params):
    parser.error("Todos los parámetros de --beta y --gamma deben ser finitos")

if any(value < 0 for value in beta_params + gamma_params):
    parser.error("Todos los parámetros de --beta y --gamma deben ser mayores o iguales a 0")

if args.check_interval <= 0:
    parser.error("--check_interval debe ser mayor a 0")

if args.state_min >= args.state_max:
    parser.error("--state_min debe ser menor que --state_max")

############################## FUNCIÓN PARA AGREGAR UNA DIMENSIÓN ##################    #############################

def create_batch(array_base, n_batch):
    # Se repite array_base n_batch veces en un bloque contiguo
    return cp.tile(array_base[cp.newaxis, :, :], (n_batch, 1, 1)).copy()


def get_state_stats(state):
    return float(cp.min(state)), float(cp.max(state))


def check_finite_and_range(name, state, lower_bound, upper_bound, step):
    finite_mask = cp.isfinite(state)
    if not bool(cp.all(finite_mask).item()):
        raise RuntimeError(f"Inestabilidad detectada en paso {step}: {name} contiene NaN o Inf")

    state_min, state_max = get_state_stats(state)
    if state_min < lower_bound or state_max > upper_bound:
        raise RuntimeError(
            f"Inestabilidad detectada en paso {step}: {name} fuera de rango [{lower_bound}, {upper_bound}] "
            f"(min={state_min:.6e}, max={state_max:.6e})"
        )

datos = preprocesar_datos()
vegetacion = datos["vegetacion"]
wx = datos["wx"]
wy = datos["wy"]
h_dx_mapa = datos["h_dx"]
h_dy_mapa = datos["h_dy"]

# Obtener dimensiones del mapa
ny, nx = vegetacion.shape  # Usamos cualquier mapa para obtener las dimensiones

for x_value, y_value in zip(ignition_x, ignition_y):
    if x_value < 0 or x_value >= nx or y_value < 0 or y_value >= ny:
        parser.error(
            f"Punto de ignición fuera del mapa: ({x_value}, {y_value}) con límites x=[0,{nx-1}] y=[0,{ny-1}]"
        )

############################## PARÁMETROS DEL INCENDIO DE REFERENCIA ###############################################

# Tamaño de cada celda
d = cp.float32(args.d) # metros
# Paso temporal
dt = cp.float32(args.dt) # horas
# Coeficiente de difusión
D_value = cp.float32(args.D) # metros^2 / hora
# Constante A adimensional de viento
A_value = cp.float32(args.A)
# Constante B de pendiente
B_value = cp.float32(args.B) # m/h

# Crear mapas de beta y gamma según tipo de vegetación
veg_types = cp.array([3, 4, 5, 6, 7], dtype=cp.int32)
beta_veg = cp.zeros_like(vegetacion, dtype=cp.float32)
gamma = cp.zeros_like(vegetacion, dtype=cp.float32)
# Asignar beta_veg según el tipo de vegetación
for j, veg_type in enumerate(veg_types):
    mask = (vegetacion == veg_type)
    beta_veg = cp.where(mask, beta_params[j], beta_veg)
    gamma = cp.where(mask, gamma_params[j], gamma)

# Poner beta y gamma en cero para vegetación 0, 1 y 2
mask_no_veg = (vegetacion == 0) | (vegetacion == 1) | (vegetacion == 2)
beta_veg = cp.where(mask_no_veg, 0.0, beta_veg)
gamma = cp.where(mask_no_veg, 0.0, gamma)

print(f'Valores de beta: {beta_params}')
print(f'Valores de gamma: {gamma_params}')
print(f'Chequeo numérico cada {args.check_interval} pasos, rango permitido [{args.state_min}, {args.state_max}]')

# Suavizar los mapas de beta y gamma
beta_veg = cupyx.scipy.ndimage.gaussian_filter(beta_veg, sigma=10.0)
gamma = cupyx.scipy.ndimage.gaussian_filter(gamma, sigma=10.0)

n_batch = 1

D = cp.full((n_batch), D_value, dtype=cp.float32)
A = cp.full((n_batch), A_value, dtype=cp.float32)
B = cp.full((n_batch), B_value, dtype=cp.float32)

############################## INCENDIO DE REFERENCIA ###############################################

# Población inicial de susceptibles e infectados
S_batch = cp.ones((n_batch, ny, nx), dtype=cp.float32)
I_batch = cp.zeros_like(S_batch)
R_batch = cp.zeros_like(S_batch)

S_batch = cp.where(vegetacion <= 2, 0, S_batch)  # Celdas no vegetadas no son susceptibles

print(f'Se cumple la condición de Courant para el término advectivo: {courant_batch(dt/2, A, B, d, wx, wy, h_dx_mapa, h_dy_mapa)}')

# Coordenadas del punto de ignición
x_ignicion = cp.array(ignition_x, dtype=cp.int32)
y_ignicion = cp.array(ignition_y, dtype=cp.int32)

S_batch[:, y_ignicion, x_ignicion] = 0
I_batch[:, y_ignicion, x_ignicion] = 1

# Definir arrays de estado
S_new_batch = cp.empty_like(S_batch)
I_new_batch = cp.empty_like(I_batch)
R_new_batch = cp.empty_like(R_batch)

beta_veg_batch = create_batch(beta_veg, n_batch)
gamma_batch = create_batch(gamma, n_batch)

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

# Iterar sobre las simulaciones
for t in range(num_steps):
    spread_infection_adi(S_batch, I_batch, R_batch, S_new_batch, I_new_batch, R_new_batch, dt, d, beta_veg_batch, gamma_batch, D, wx, wy, h_dx_mapa, h_dy_mapa, A, B, vegetacion)

    # Swap de buffers (intercambiar referencias en lugar de crear nuevos arrays)
    S_batch, S_new_batch = S_new_batch, S_batch
    I_batch, I_new_batch = I_new_batch, I_batch
    R_batch, R_new_batch = R_new_batch, R_batch

    if (t + 1) % args.check_interval == 0 or (t + 1) == num_steps:
        check_finite_and_range("S", S_batch, args.state_min, args.state_max, t + 1)
        check_finite_and_range("I", I_batch, args.state_min, args.state_max, t + 1)
        check_finite_and_range("R", R_batch, args.state_min, args.state_max, t + 1)

end.record()  # Marca el final en GPU
end.synchronize() # Sincroniza y mide el tiempo

cp.save(f"R_referencia_{exp}.npy", R_new_batch)

gpu_time = cp.cuda.get_elapsed_time(start, end)  # Tiempo en milisegundos
print(f"Tiempo en GPU: {gpu_time:.3f} ms")

print(f'Numero de celdas quemadas: {cp.sum(R_new_batch > 0.001)}')

############################## VISUALIZACIÓN DE LOS MAPAS ###############################################

if visualizar_mapas:
    import matplotlib.pyplot as plt
    import scienceplots
    
    plt.style.use(['science', 'ieee'])

    # Definir los nuevos colores para los valores del archivo (0 a 7)
    vegetation_colors = np.array([
        [255, 0, 255],      # 0: NODATA (magenta)
        [199, 209, 207],    # 1: Sin combustible (gris claro)
        [0, 0, 255],        # 2: Lagos (azul)
        [0, 117, 0],        # 3: Bosque A (verde oscuro)
        [50, 200, 10],      # 4: Bosque B (verde brillante)
        [150, 0, 150],      # 5: Bosque I (morado)
        [122, 127, 50],     # 6: Pastizal (verde oliva)
        [0, 196, 83]        # 7: Arbustal (verde intenso)
    ]) / 255.0  # Escalar los valores RGB al rango [0, 1]

    # Mapear los valores de vegetación a colores RGB
    vegetation = vegetation_colors[vegetacion.get().astype(int)]

    # Crear visualización de vectores de viento sobre el mapa de vegetación
    fig, ax = plt.subplots(figsize=(4,2))

    # Transferencia del mapa a CPU
    R_cpu = np.squeeze(R_new_batch.get())

    # Mostrar el mapa de vegetación como fondo
    terrain_rgb = (1 - np.clip(3*R_cpu[..., None], 0, 1)) * vegetation + np.clip(3*R_cpu[..., None], 0, 1) * np.array([1.0, 0.0, 0.0])
    im = ax.imshow(terrain_rgb, interpolation='nearest', origin='lower')

    x_ticks = np.arange(0, nx, 200)  # Cada 200 celdas en el eje X
    y_ticks = np.arange(0, ny, 200)  # Cada 200 celdas en el eje Y
    x_labels = (x_ticks * d) / 1000  # Convertir a kilómetros
    y_labels = (y_ticks * d) / 1000

    # Configurar ejes y etiquetas
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{x:.1f}" for x in x_labels], fontsize=4)
    ax.set_yticklabels([f"{y:.1f}" for y in y_labels], fontsize=4)
    ax.set_xlabel("X (km)", fontsize=4)
    ax.set_ylabel("Y (km)", fontsize=4)

    ax.scatter(ignition_x, ignition_y, color='red', marker='*', s=10, edgecolors='black', linewidths=0.5)

    plt.tight_layout()
    plt.savefig(f'R_referencia_{exp}_map.pdf', transparent=True, dpi=600, bbox_inches='tight')
    plt.show()
import cupy as cp # type: ignore
import time
from config import d, dt, cota
from algoritmo import genetic_algorithm
from data_context import DataContext
import argparse

print(f"Placa Gráfica: {cp.cuda.runtime.getDeviceProperties(0)['name']}")

# Obtengo la variable por línea de comando
parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=int, default=1, help="Número de experimento")
parser.add_argument("--pretrained", type=str, default=None, help="Ruta al archivo preentrenado")
parser.add_argument("--start_gen", type=int, default=0, help="Generación desde la que entrenar")
parser.add_argument("--ruta_incendio_referencia", type=str, default=None, help="Ruta al incendio de referencia")
parser.add_argument("--incendio_real", action="store_true", help="Si se ajusta un incendio real = True")
parser.add_argument("--num_steps", type=int, default=500, help="Número de pasos a simular")
parser.add_argument("--tamano_poblacion", type=int, default=10000, help="Tamaño de la población")
parser.add_argument("--num_generaciones", type=int, default=20, help="Número de generaciones a ejecutar por el AG")
parser.add_argument("--batch_size", type=int, default=5, help="Tamaño del batch, número de simulaciones realizadas en simultáneo")
args = parser.parse_args()

############################## CARGADO DE MAPAS #######################################################

ctx = DataContext().load()
wx = ctx.wx
wy = ctx.wy
h_dx_mapa = ctx.h_dx
h_dy_mapa = ctx.h_dy

############################## PARSEO DE PARÁMETROS ####################################

archivo_preentrenado = args.pretrained
generacion_preentranada = args.start_gen
exp = args.exp
ruta_incendio_referencia = args.ruta_incendio_referencia
incendio_real = args.incendio_real
num_steps = args.num_steps
tamano_poblacion = args.tamano_poblacion
generaciones = args.num_generaciones
batch_size = args.batch_size 

############################## CONDICIÓN DE COURANT PARA LOS TÉRMINOS DIFUSIVOS Y ADVECTIVOS ####################################

A_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(wx**2+wy**2)))) # constante de viento
B_max = float(d / (cp.sqrt(2)*dt/2*cp.max(cp.sqrt(h_dx_mapa**2+h_dy_mapa**2)))) # constante de pendiente

############################## DISEÑO DE EXPERIMENTOS ##########################################

limite_parametros_base = [
    (0.01, 100.0),          # D
    (0.0, A_max * cota),    # A
    (0.0, B_max * cota)     # B
]

print(f"Corriendo el experimento {exp}")

# Configuración de tipos de vegetación
veg_types = None
num_combustibles = 5

if exp == 1:
    ajustar_beta_gamma = False
    ajustar_ignicion = True

    limite_ignicion = [(300, 720), (400, 800)]
    limite_parametros = limite_parametros_base + limite_ignicion

    # Beta y gamma fijos para cada tipo de vegetación (en orden: tipo 3, 4, 5, 6, 7)
    beta_fijo = [0.91, 0.72, 1.38, 1.94, 0.75]
    gamma_fijo = [0.5, 0.38, 0.84, 0.45, 0.14]

elif exp == 2:
    ajustar_beta_gamma = True
    ajustar_ignicion = True

    limite_ignicion = [(300, 720), (400, 800)]
    limite_beta = [(0.01, 2.0)]
    limite_gamma = [(0.01, 0.9)]
    limite_parametros = limite_parametros_base + limite_ignicion + limite_beta + limite_gamma

elif exp == 3:
    ajustar_beta_gamma = True
    ajustar_ignicion = False
    
    num_combustibles = 4 if incendio_real else 5

    limite_beta = [(0.01, 5.0)] * num_combustibles
    limite_gamma = [(0.01, 5.0)] * num_combustibles
    
    limite_parametros = limite_parametros_base + limite_beta + limite_gamma

    if incendio_real:
        ignicion_fija_x = [475, 565]
        ignicion_fija_y = [550, 530]
        
        # En el caso del incendio del Steffen-Martin, el incendio no pasó por zonas con bosque insertado
        veg_types = [3, 5, 6, 7]
    else:
        ignicion_fija_x = [1130, 1300, 620]
        ignicion_fija_y = [290, 150, 280]

else:
    raise ValueError(f"Experimento {exp} no está definido")

# Mostrar configuración
print(f"\n{'='*60}")
print(f"CONFIGURACIÓN DEL EXPERIMENTO {exp}")
print(f"{'='*60}")
print(f"Número de combustibles: {num_combustibles}")
print(f"Tipos de vegetación: {'Autodetección' if veg_types is None else veg_types}")
print(f"Ajustar beta/gamma: {ajustar_beta_gamma}")
print(f"Ajustar ignición: {ajustar_ignicion}")
print(f"Tamaño del batch: {batch_size}")
print(f"Número de pasos a simular: {num_steps}")
print(f"Tamaño de la población en individuos: {tamano_poblacion}")
print(f"Número de generaciones a ejecutar: {generaciones}")
print(f"{'='*60}\n")

############################## EJECUCIÓN DEL ALGORITMO ###############################################

# Sincronizar antes de empezar a medir el tiempo
cp.cuda.Stream.null.synchronize()
start_time = time.time()

resultados = genetic_algorithm(
    tamano_poblacion=tamano_poblacion,
    generaciones=generaciones,
    limite_parametros=limite_parametros,
    ruta_incendio_referencia=ruta_incendio_referencia,
    ctx=ctx,
    archivo_preentrenado=archivo_preentrenado,
    generacion_preentrenada=generacion_preentranada,
    num_steps=num_steps,
    batch_size=batch_size,
    num_combustibles=num_combustibles,
    ajustar_beta_gamma=ajustar_beta_gamma,
    beta_fijo=beta_fijo if not ajustar_beta_gamma else None,
    gamma_fijo=gamma_fijo if not ajustar_beta_gamma else None,
    ajustar_ignicion=ajustar_ignicion,
    ignicion_fija_x=ignicion_fija_x if not ajustar_ignicion else None,
    ignicion_fija_y=ignicion_fija_y if not ajustar_ignicion else None,
    veg_types=veg_types
)

# Sincronizar después de completar la ejecución
cp.cuda.Stream.null.synchronize()
end_time = time.time()

print(f"Tiempo de ejecución en GPU: {end_time - start_time} segundos")
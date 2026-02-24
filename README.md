# 🔥 Repositorio de la Tesis de Maestría en Ciencias Físicas del Instituto Balseiro: Simulaciones computacionales y visualización de la propagación de incendios forestales en la región patagónica

## Maestrando: Lic. Lucas Becerra
## Directora: Dra. Karina Laneri
## Co-directora: Dra. Mónica Malen Denham

Este repositorio contiene el material correspondiente a la Tesis de Maestría en Ciencias Físicas del Instituto Balseiro, titulada 'Simulaciones computacionales y visualización de la propagación de incendios forestales en la región patagónica', dirigida por la Dra. Karina Laneri y por la Dra. Mónica Malen Denham.

## 📁 Estructura del repositorio

- `modelo_rdc.py` — Implementación del modelo de reacción-difusión-convección.

La implementación numérica del modelo RDC fue realizada mediante descomposición de operadores (*operator splitting*) en la que los términos de reacción y convección fueron discretizados mediante un esquema de Euler explícito y el término de difusión mediante el esquema implícito *alternating direction implicit* (ADI). La descripción matemática de los métodos se encuentra en el Capítulo 2 de la tesis.

- `fuego_referencia.py` — Simulación de referencia para comparación entre métodos.

El programa `fuego_referencia.py` permite realizar simulaciones de referencia con parámetros configurables. Se encuentran configurados los tres experimentos sintéticos descritos en la tesis y utilizados para recuperar los parámetros.

**Parámetros disponibles:**
- `--exp` (int, default=1): Número de experimento (1, 2 o 3)
- `--num_steps` (int, default=500): Número de pasos de simulación
- `--d` (float, default=30.0): Tamaño de celda en metros
- `--dt` (float, default=0.5): Paso temporal en horas
- `--D` (float, default=10.0): Coeficiente de difusión
- `--A` (float, default=1e-4): Constante adimensional de viento
- `--B` (float, default=15.0): Constante de pendiente
- `--beta` (float list): Parámetros de ignición por tipo de vegetación (tipos 3,4,5,6,7)
- `--gamma` (float list): Parámetros de extinción por tipo de vegetación
- `--ignition_x` (int list): Coordenadas X del/los punto(s) de ignición
- `--ignition_y` (int list): Coordenadas Y del/los punto(s) de ignición
- `--check_interval` (int, default=10): Cada cuántos pasos chequear estabilidad numérica
- `--state_min` (float, default=-1000): Valor mínimo permitido en S,I,R
- `--state_max` (float, default=1000): Valor máximo permitido en S,I,R
- `--visualizar_mapas`: Generar visualización de los resultados

**Ejemplos:**

```bash
# Experimento 1 con visualización
python fuego_referencia.py --exp 1 --visualizar_mapas

# Experimento 3 con parámetros personalizados
python fuego_referencia.py --exp 3 --num_steps 800 --D 12 --A 0.0002 --B 20 \
  --beta 1.2 1.1 1.0 0.9 0.8 --gamma 0.4 0.4 0.5 0.5 0.6 \
  --ignition_x 1130 1300 620 --ignition_y 290 150 280 --visualizar_mapas
```

**Validaciones integradas:**
El programa valida automáticamente:
- Límites físicos de los parámetros (D, A, B ≥ 0)
- Consistencia de beta/gamma (misma cantidad de valores)
- Puntos de ignición dentro del dominio del mapa
- Detecta y notifica valores NaN/Inf durante la simulación

- `mapas/`
  - `mapas_steffen_martin` - Contiene los mapas raster utilizados
  - `io_mapas.py` - Funciones de lectura y procesado de mapas

- `genetico/` — Contiene los scripts en Python para ejecutar los métodos de fuerza bruta y el algoritmo genético
  - `algoritmo.py` — Itera el algoritmo genético utilizando los operadores evolutivos
  - `config.py` — Contiene valores como el tamaño del paso temporal y la distancia entre celdas
  - `fitness.py` — Clase `FitnessEvaluator` que realiza simulaciones batch con los siguientes puntos clave:
    - Evaluación en paralelo de múltiples configuraciones de parámetros
    - Validaciones de condición de Courant, puntos de ignición y parámetros beta/gamma
  - `lectura_datos.py` — Carga una población entrenada y la guarda luego de una corrida del algoritmo genético
  - `operadores_geneticos.py` — Implementación de los operadores de selección, cruce y mutación
  - `main.py` — Ejecuta el algoritmo genético. Requiere el mapa de referencia generado por `fuego_referencia.py`.

    **Parámetros completos:**
    
    | Parámetro | Tipo | Default | Descripción |
    |-----------|------|---------|-------------|
    | `--exp` | int | 1 | Número de experimento (1, 2 o 3). Define el espacio de búsqueda y parámetros a ajustar |
    | `--num_steps` | int | 500 | Número de pasos de simulación en cada evaluación de fitness |
    | `--tamano_poblacion` | int | 10000 | Cantidad de individuos en la población del AG. Mayor → búsqueda más exhaustiva pero más lenta |
    | `--num_generaciones` | int | 20 | Número de generaciones a evolucionar. Más generaciones = convergencia más fina |
    | `--batch_size` | int | 5 | Simulaciones ejecutadas **en paralelo en GPU**. Aumentar mejora velocidad, pero consume más memoria GPU |
    | `--pretrained` | str | None | Ruta a un archivo de población preentrenada (.csv) para continuar optimización |
    | `--start_gen` | int | 0 | Generación desde la que continuar (solo si `--pretrained` está definido) |
    | `--ruta_incendio_referencia` | str | None | Ruta al archivo `.npy` del incendio de referencia (ej: `R_referencia_1.npy`) |
    | `--incendio_real` | flag | False | Si se pasa, ajusta un incendio real en lugar de sintético (solo Exp3) |
    | `--verbose` | flag | False | Activa mensajes de debug detallados (rescates de NaN, convergencia, etc.) |
    
    
    **Ejemplo básico:**
    ```bash
    python main.py --exp 1 --ruta_incendio_referencia '../R_referencia_1.npy' --verbose
    ```

  - `fuerza_bruta.py` — Exploración exhaustiva del espacio de parámetros (*brute force*).

    Para ejecutar:

    ```bash
    python fuerza_bruta.py --exp 1
    ```

- `pinns/` — Entrenamiento de redes neuronales informadas por la física (Physics-Informed Neural Networks, PINNs).
  - `train_pinn.py` - Modelo de PINN
  - `pinns_sir.py` - Entrenamiento de la PINN. Para entrenar una PINN desde el directorio PINNS, hay que ejecutar:
 
  ```bash
  python pinns_sir.py
  ```

- `.gitignore` — Ignora archivos temporales y entornos virtuales, de Python.
- `README.md` — Este archivo.

## ⚙️ Dependencias y requerimientos

El código de este repositorio fue desarrollado en **Python** y está orientado a la simulación numérica y análisis computacional de incendios forestales, con énfasis en ejecución acelerada por GPU.

### 📦 Dependencias principales

Las principales bibliotecas utilizadas son:

- **NumPy** — Operaciones numéricas y manejo de arreglos.
- **SciPy** — Métodos numéricos y resolución de sistemas.
- **Matplotlib** — Visualización de resultados.
- **CuPy** — Computación acelerada por GPU compatible con CUDA.
- **PyTorch** — Implementación y entrenamiento de redes neuronales informadas por la física (PINNs).
- **Rasterio** — Lectura y manejo de mapas raster geoespaciales.

Algunas dependencias pueden ser opcionales dependiendo del módulo que se desee ejecutar.

### 🚀 Requerimientos de GPU

- GPU compatible con **CUDA** (NVIDIA).
- Drivers de NVIDIA y versión de CUDA compatibles con la versión de **CuPy** instalada.
- Para el entrenamiento de PINNs, se recomienda disponer de al menos **8 GB de memoria de GPU**.

## 🔧 Flujo de Trabajo Típico

### 1. Generar Incendio de Referencia
```bash
python fuego_referencia.py --exp 1 --num_steps 500 --visualizar_mapas
```
Esto genera `R_referencia_1.npy` que será utilizado como objetivo para la optimización.

### 2. Optimizar Parámetros con Algoritmo Genético
```bash
python Genetico/main.py --exp 1 --num_generaciones 20 --batch_size 10 --ruta_incendio_referencia '../R_referencia_1.npy' --verbose
```
El algoritmo genético busca ajustar los parámetros del modelo para reproducir el incendio de referencia.

### Alternativa: Exploración exhaustiva
```bash
python Genetico/fuerza_bruta.py --exp 1
```
Prueba todas las combinaciones de parámetros dentro de los límites especificados (más costoso computacionalmente).

### 3. Visualizar y Analizar
Los resultados se guardan en `Genetico/resultados/` con un archivo CSV por generación.

## 📝 Notas sobre los Parámetros

- **Beta (β)**: Tasa de ignición por tipo de vegetación.
- **Gamma (γ)**: Tasa de extinción por tipo de vegetación. Deben cumplir γ < β para estabilidad física
- **D**: Coeficiente de difusión (propagación base).
- **A**: Constante adimensional de viento. Amplifica el efecto de los vientos en la propagación
- **B**: Constante de pendiente. Amplifica el efecto de la topografía en la propagación
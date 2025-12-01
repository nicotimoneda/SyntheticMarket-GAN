# SyntheticMarket-GAN

Este proyecto implementa una Generative Adversarial Network (GAN) para generar datos sintéticos del mercado de valores (específicamente AAPL).

## Guía de Inicio Rápido (Onboarding)

### 1. Configuración del Entorno

El proyecto utiliza **uv** para una gestión de dependencias rápida y fiable.

* **Instalar uv** (si no lo tienes):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

* **Instalar dependencias**:

    ```bash
    uv sync
    ```

    Esto creará el entorno virtual e instalará todas las librerías necesarias definidas en `pyproject.toml`.

### 2. Generación de Datos

Antes de entrenar, necesitas descargar y procesar los datos históricos.

* **Ejecutar el script de datos**:

    ```bash
    uv run python src/make_dataset.py
    ```

  * Descarga datos de AAPL desde Yahoo Finance.
  * Escala los precios (MinMax) y guarda el resultado en `data/processed/AAPL_scaled.csv`.
  * Guarda el escalador en `data/processed/AAPL_scaler.pkl`.

### 3. Entrenamiento del Modelo (WGAN-GP)

El núcleo del proyecto es la Wasserstein GAN con Gradient Penalty.

* **Ejecutar el notebook**:
    Abre y ejecuta todas las celdas de `notebooks/06_WGAN_GP.ipynb`.
  * Carga los datos procesados.
  * Entrena el Generador y el Crítico.
  * Muestra métricas de evaluación (Pérdida, PCA, t-SNE).
  * Guarda el modelo entrenado en `models/generator_wgan.pth`.

## Estructura del Proyecto

* **`src/`**: Código fuente modular.
  * `make_dataset.py`: Orquestador de datos.
  * `data_loader.py`: Descarga de datos.
  * `preprocessing.py`: Lógica de escalado y secuencias.
* **`notebooks/`**: Experimentación.
  * `06_WGAN_GP.ipynb`: **Notebook Principal**.
* **`data/`**: Almacenamiento de datos (`raw` y `processed`).
* **`models/`**: Modelos entrenados (`.pth`).
* **`pyproject.toml`**: Definición de dependencias.
* **`uv.lock`**: Archivo de bloqueo para reproducibilidad exacta.

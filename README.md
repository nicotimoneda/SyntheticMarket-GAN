# 📈 SyntheticMarket-GAN

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Generación de Datos Financieros Sintéticos de Alta Fidelidad usando WGAN-GP.**

Este proyecto implementa una **Wasserstein GAN con Gradient Penalty (WGAN-GP)** diseñada para aprender y reproducir la dinámica compleja de los mercados financieros (específicamente acciones como AAPL). A diferencia de las GANs tradicionales, esta arquitectura ofrece una estabilidad de entrenamiento superior y evita el colapso de modos, generando series temporales indistinguibles de las reales.

---

## 🚀 Características Clave

* **Arquitectura Robusta**: Implementación de **WGAN-GP** para garantizar la convergencia y estabilidad del entrenamiento.
* **Deep Learning**: Uso de redes **LSTM** tanto en el Generador como en el Crítico para capturar dependencias temporales a largo plazo.
* **Pipeline de Datos Automatizado**: Scripts modulares para la descarga, limpieza y escalado de datos financieros en tiempo real.
* **Gestión de Entorno Moderna**: Uso de `uv` para una gestión de dependencias ultrarrápida y reproducible.
* **Evaluación Rigurosa**: Análisis de calidad mediante PCA, t-SNE y métricas estadísticas.

## 🛠️ Instalación

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para la gestión de paquetes.

1. **Clonar el repositorio**:

    ```bash
    git clone https://github.com/tu-usuario/SyntheticMarket-GAN.git
    cd SyntheticMarket-GAN
    ```

2. **Configurar el entorno**:

    ```bash
    # Instalar dependencias y crear entorno virtual automáticamente
    uv sync
    ```

## 📊 Uso

### 1. Preparación de Datos

Descarga y preprocesa los datos históricos más recientes de AAPL:

```bash
uv run python src/make_dataset.py
```

*Esto generará `data/processed/AAPL_scaled.csv` listo para el entrenamiento.*

### 2. Entrenamiento del Modelo

Para entrenar la GAN y visualizar los resultados en tiempo real:

1. Abre el notebook principal:

    ```bash
    uv run jupyter notebook notebooks/06_WGAN_GP.ipynb
    ```

2. Ejecuta todas las celdas para iniciar el entrenamiento del WGAN-GP.
3. El modelo entrenado se guardará automáticamente en `models/generator_wgan.pth`.

## 📂 Estructura del Proyecto

```text
SyntheticMarket-GAN/
├── data/                  # Almacenamiento de datos
│   ├── raw/               # Datos crudos descargados (Yahoo Finance)
│   └── processed/         # Datos escalados y listos para ML
├── models/                # Checkpoints de modelos entrenados (.pth)
├── notebooks/             # Entorno de experimentación
│   └── 06_WGAN_GP.ipynb   # ⭐️ Notebook Principal (WGAN-GP)
├── src/                   # Código fuente modular
│   ├── data_loader.py     # Módulo de descarga de datos
│   ├── make_dataset.py    # Script de orquestación de datos
│   └── preprocessing.py   # Lógica de transformación y secuencias
├── pyproject.toml         # Definición de dependencias
└── uv.lock                # Lockfile para reproducibilidad exacta
```

## 📈 Resultados

El modelo es capaz de generar secuencias de precios que replican las propiedades estadísticas de los datos reales. Las visualizaciones de **t-SNE** y **PCA** incluidas en el notebook demuestran una superposición significativa entre las distribuciones reales y sintéticas.

## 🤝 Contribución

Las contribuciones son bienvenidas. Si tienes ideas para mejorar la arquitectura o añadir nuevos activos, no dudes en abrir un *issue* o enviar un *pull request*.

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---
*Desarrollado por Nicolás Timoneda*

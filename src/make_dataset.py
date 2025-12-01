import os
import pandas as pd
import joblib
from data_loader import download_data
from preprocessing import DataPreprocessor


def main():
    # Configuración
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2025-11-29"
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"

    # Asegurar que existan directorios
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Obtener datos crudos
    csv_filename = f"{TICKER}_{START_DATE}_{END_DATE}.csv"
    csv_path = os.path.join(RAW_DIR, csv_filename)

    if os.path.exists(csv_path):
        print(f"Cargando datos crudos desde: {csv_path}")
        df = pd.read_csv(
            csv_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True
        )
        # Ajuste para yfinance headers complejos si es necesario,
        # pero asumimos que el usuario ya tiene el csv correcto o data_loader lo baja bien.
        # Si data_loader baja con multi-index headers, hay que tener cuidado.
        # Revisando data_loader.py, usa yf.download.
    else:
        print(f"Descargando datos para {TICKER}...")
        df = download_data(TICKER, START_DATE, END_DATE, save_dir=RAW_DIR)

    # Limpieza básica si es necesario (e.g. dropna)
    df = df.dropna()

    # 2. Preprocesar (Escalado)
    print("Preprocesando datos...")
    preprocessor = DataPreprocessor()
    # Ajustamos el scaler con 'Close'
    # Nota: yfinance a veces devuelve multi-index columns.
    # Vamos a asegurar que accedemos a 'Close' correctamente.

    if isinstance(df.columns, pd.MultiIndex):
        # Si es multi-index, intentamos aplanar o seleccionar el ticker
        try:
            data_to_scale = df["Close"]
        except KeyError:
            # Fallback si la estructura es diferente
            print("Estructura de columnas compleja, intentando simplificar...")
            df.columns = df.columns.get_level_values(0)
            data_to_scale = df["Close"]
    else:
        data_to_scale = df[
            ["Close"]
        ]  # Doble corchete para mantener DataFrame si es necesario por el preprocessor

    # El preprocessor espera un DataFrame y una lista de columnas
    # Modificamos un poco la lógica para usar el preprocessor existente
    # preprocessor.fit_transform(data, columns)

    # Para simplificar y asegurar compatibilidad, pasamos el DF completo y la columna 'Close'
    # Pero primero aseguremonos que 'Close' es accesible y única
    if "Close" not in df.columns and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    scaled_data = preprocessor.fit_transform(df, ["Close"])

    # 3. Guardar datos procesados y el scaler
    processed_data_path = os.path.join(PROCESSED_DIR, f"{TICKER}_scaled.csv")
    scaler_path = os.path.join(PROCESSED_DIR, f"{TICKER}_scaler.pkl")

    # Guardar como CSV (sin índice de fecha para simplificar la carga en tensores, o con fecha si se prefiere)
    # Para GANs, solemos querer solo los valores, pero guardar la fecha es útil para referencias.
    # Guardaremos un DF con fecha y valor escalado.

    df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=["Close_Scaled"])
    df_scaled.to_csv(processed_data_path)

    # Guardar el scaler para poder invertir la transformación después
    joblib.dump(preprocessor.scaler, scaler_path)

    print(f"Datos procesados guardados en: {processed_data_path}")
    print(f"Scaler guardado en: {scaler_path}")


if __name__ == "__main__":
    main()

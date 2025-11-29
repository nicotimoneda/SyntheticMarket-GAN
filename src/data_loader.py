import os
import yfinance as yf
import pandas as pd
from datetime import datetime


def download_data(
    ticker: str, start_date: str, end_date: str, save_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Descarga datos históricos de acciones usando yfinance y los guarda en un archivo CSV.

    Args:
        ticker (str): El símbolo del ticker de la acción (ej. 'AAPL').
        start_date (str): La fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): La fecha de fin en formato 'YYYY-MM-DD'.
        save_dir (str): El directorio para guardar el archivo CSV. Por defecto es 'data/raw'.

    Returns:
        pd.DataFrame: El DataFrame descargado.
    """
    # Crear el directorio si no existe
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directorio creado: {save_dir}")

    print(f"Descargando datos para {ticker} desde {start_date} hasta {end_date}...")

    # Descargar datos
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            print(f"No se encontraron datos para {ticker} en el rango especificado.")
            return pd.DataFrame()

        # Construir nombre de archivo
        filename = f"{ticker}_{start_date}_{end_date}.csv"
        filepath = os.path.join(save_dir, filename)

        # Guardar en CSV
        df.to_csv(filepath)
        print(f"Datos guardados en {filepath}")

        return df

    except Exception as e:
        print(f"Ocurrió un error al descargar datos para {ticker}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Probar la función con AAPL
    ticker_symbol = "AAPL"
    start = "2015-01-01"
    end = datetime.today().strftime("%Y-%m-%d")

    df_aapl = download_data(ticker_symbol, start, end)

    if not df_aapl.empty:
        print("\nEncabezado de los datos descargados:")
        print(df_aapl.head())
        print("\nForma de los datos descargados:")
        print(df_aapl.shape)

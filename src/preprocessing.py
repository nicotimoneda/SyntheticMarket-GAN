import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


class DataPreprocessor:
    """
    Clase para preprocesar datos financieros para modelos de Deep Learning.
    """

    def __init__(self, feature_range: Tuple[int, int] = (0, 1)):
        """
        Inicializa el preprocesador.

        Args:
            feature_range (Tuple[int, int]): Rango para la normalización MinMaxScaler.
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """
        Ajusta el escalador y transforma los datos seleccionados.

        Args:
            data (pd.DataFrame): DataFrame con los datos originales.
            columns (List[str]): Lista de columnas a escalar.

        Returns:
            np.ndarray: Datos escalados.
        """
        return self.scaler.fit_transform(data[columns])

    def transform(self, data: pd.DataFrame, columns: List[str]) -> np.ndarray:
        """
        Transforma los datos usando el escalador ya ajustado.

        Args:
            data (pd.DataFrame): DataFrame con los datos originales.
            columns (List[str]): Lista de columnas a escalar.

        Returns:
            np.ndarray: Datos escalados.
        """
        return self.scaler.transform(data[columns])

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Invierte la transformación de escala.

        Args:
            data (np.ndarray): Datos escalados.

        Returns:
            np.ndarray: Datos en su escala original.
        """
        return self.scaler.inverse_transform(data)


def create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Crea secuencias de ventanas deslizantes a partir de los datos.

    Args:
        data (np.ndarray): Los datos escalados.
        seq_len (int): La longitud de la secuencia.

    Returns:
        np.ndarray: Array de secuencias con forma (n_muestras, seq_len, n_features).
    """
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i : i + seq_len]
        sequences.append(seq)

    return np.array(sequences)

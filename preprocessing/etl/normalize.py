from typing import Optional
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize(
    data: pd.DataFrame,
    scaler: Optional[MinMaxScaler] = None,
    data_filename: Optional[str] = None,
    scaler_filename: Optional[str] = None
) -> pd.DataFrame:
    data_numpy = data.to_numpy()
    if len(data.shape) != 2:
        raise ValueError("Data should be 2 dimentional")
    if scaler:
        scaled_data = scaler.transform(data_numpy)
    else:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data_numpy)
    result = pd.DataFrame(
        data=scaled_data,
        columns=data.columns
    )
    if data_filename:
        data_filename = data_filename if data_filename.endswith(".csv") else data_filename + ".csv" 
        result.to_csv(data_filename)
    if scaler_filename:
        scaler_filename = scaler_filename if scaler_filename.endswith(".pkl") else scaler_filename + ".pkl" 
        joblib.dump(scaler, scaler_filename)
    return result

from typing import Tuple, Union
import os
import joblib

import pandas as pd
import numpy as np
from finta import TA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader

from preprocessing.etl import normalize
from clustering.dataset_model import CnnDataset


class DataPrepareETL: 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = [
        "Close",
        "prediction",
        "cluster"
    ]
    batch_size = 64

    def __init__(self, data: Union[pd.DataFrame, str], scaler: Union[MinMaxScaler, str] = None):
        self.df = self._load_data(data) if isinstance(data, str) else data
        self._check_data(self.df)
        self.scaler = self._load_scaler(scaler) if isinstance(scaler, str) else scaler
        self.y_column_idx = self.features.index("Close") - 1

    def _load_data(self, data_filename: str):
        data = pd.read_csv(data_filename, parse_dates=True, index_col="time_open")
        return data

    def _check_data(self, df: pd.DataFrame) -> bool:
        for col in self.features:
            if col not in df.columns:
                raise ValueError(f"Dataset should include these features: {self.features}")
        return True

    def _load_scaler(self, scaler_filename: str):
        scaler = joblib.load(scaler_filename)
        return scaler
    
    def prepare(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature Extending
        self.df["RSI"] = TA.RSI(self.df)
        self.df["OBV"] = TA.OBV(self.df)
        self.df["prediction"] = (self.df["prediction"] - self.df["Close"]) / self.df["Close"]
        self.df.dropna(inplace=True)
        self.features.remove("Close")
        self.features.extend(["RSI", "OBV"])
        self.df["price_change"] = 100 * (self.df["Close"].shift(-1) - self.df["Close"]) / self.df["Close"]
        # Normalization
        normalized = normalize(self.df[self.features], self.scaler)
        normalized["price_change"] = self.df["price_change"].values
        return normalized
    
    def prepare_for_train(self, threshold: float, valid_size: float, test_size: float) -> Tuple[
        DataLoader,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ]:
        if not (0 < 1 - (valid_size + test_size) < 1):
            raise ValueError("train/valid/test sizes should be summarized into 1")
        valid_size = valid_size / (1 - test_size)
        data = self.prepare()
        data = data.sample(frac=1).reset_index(drop=True)
        X = data.drop("price_change", axis=1).to_numpy()
        y = np.where(
            data["price_change"].to_numpy() <= -threshold,
            0,
            np.where(data["price_change"].to_numpy() >= threshold,
                     2, 
                     1
            )
        )
        # split the data
        X_train_and_valid, X_test, y_train_and_valid, y_test = train_test_split(X, y, test_size=test_size)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_and_valid, y_train_and_valid, test_size=valid_size)
        # create dataloader for train only
        train_dataset = CnnDataset(X_train, y_train)    
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        return train_loader, (X_valid, y_valid), (X_test, y_test)


if __name__ == "__main__":
    static_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
    
    df = pd.read_csv(os.path.join(static_dir, "BTCUSDT 1H.csv"), parse_dates=True, index_col="time_open")
    scaler = joblib.load(os.path.join(static_dir, "feature_MinMaxScaler.pkl"))
    
    pipeline = DataPrepareETL(df, scaler)
    print(f"Pipeline run. device: {pipeline.device}")
    train_loader, validation_dataset, test_dataset = pipeline.prepare_for_train(0.1, .10, .10)
    
    joblib.dump(train_loader, os.path.join(static_dir, "train_dataloader.pkl"))
    joblib.dump(validation_dataset, os.path.join(static_dir, "validation_dataset.pkl"))
    joblib.dump(test_dataset, os.path.join(static_dir, "test_dataset.pkl"))
    
    for X_batch, y_batch in train_loader:
        break
    print("Train Batch Size:", X_batch.shape, y_batch.shape)
    print("Validation Set Size:", validation_dataset[0].shape, validation_dataset[1].shape)
    print("Test Set Size:", test_dataset[0].shape, test_dataset[1].shape)

from typing import Tuple, Union
import os
import joblib
from datetime import timedelta

import pandas as pd
import numpy as np
from finta import TA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader

from preprocessing.etl import normalize
from utils import TrainDataset


class DataPrepareETL: 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = [
        "cluster",
        "prediction"
    ]
    batch_size = 64

    def __init__(
        self,
        data: Union[pd.DataFrame, str],
        scaler: Union[MinMaxScaler, str] = None,
        last_year: bool = False
    ):
        self.df = self._load_data(data) if isinstance(data, str) else data
        self._check_data(self.df)
        if last_year:
            self.df = self.df[self.df.index >= (self.df.index[-1] - timedelta(days=365))]
        self.scaler = self._load_scaler(scaler) if isinstance(scaler, str) else scaler

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
        self.df["price_change"] = 100 * (self.df["Close"].shift(-1) - self.df["Close"]) / self.df["Close"]
        self.df.dropna(inplace=True)
        if "RSI" not in self.features:
            self.features.append("RSI")
        if "OBV" not in self.features:
            self.features.append("OBV")
        # Normalization
        normalized = normalize(self.df[self.features], self.scaler)
        for feature in self.features:
            self.df[feature] = normalized[feature].to_numpy()
        return self.df

import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from backtesting import Strategy

from reinforcement.agent import Agent, Model


class AgentBasedStrategy(Strategy):
    agent: Agent = Agent()
    scaler: MinMaxScaler = joblib.load("../reinforcement/static/feature_MinMaxScaler.pkl")
    features = [
        "cluster",
        "prediction",
        "RSI",
        "OBV"
    ]    
    
    def init(self):
        self.signal = self.I(
            self.agent.act_in_strategy,
            self.data.cluster,
            self.data.prediction,
            self.data.RSI,
            self.data.OBV
        )
    
    def next(self):
        if self.signal == 2:
            if self.position.is_short:
                self.position.close()
            self.buy(size=1)
        elif self.signal == 0:
            if self.position.is_long:
                self.position.close()
            self.sell(size=1)
        else:
            if self.position:
                self.position.close()

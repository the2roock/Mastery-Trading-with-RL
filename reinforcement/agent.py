import numpy as np
import pandas as pd
import torch

from .policy_network import load_trained, Model


class Agent:
    def __init__(self):
        self.model = load_trained()
        
    def act(self, state: np.ndarray, batch: bool = False) -> np.ndarray:
        state = state if batch else [state]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probabilities = torch.softmax(self.model(state_tensor), dim=1)
        actions = torch.argmax(action_probabilities, dim=1)
        return actions.numpy()
    
    def act_in_strategy(
        self,
        cluster: float, 
        prediction: float,
        rsi: float,
        obv: float
    ) -> int:
        data = pd.DataFrame({
            "cluster": cluster,
            "prediction": prediction,
            "RSI": rsi,
            "OBV": obv
        })
        state = torch.tensor(data.to_numpy(), dtype=torch.float32)
        with torch.no_grad():
            action_probabilities = torch.softmax(self.model(state), dim=1)
        actions = torch.argmax(action_probabilities, dim=1)
        return actions.numpy()
    
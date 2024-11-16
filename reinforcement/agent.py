import numpy as np
import torch

from .policy_network import load_trained


class Agent:
    def __init__(self):
        self.model = load_trained()
        
    def act(self, state: np.ndarray, batch: bool = False) -> np.ndarray:
        state = state if batch else [state]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_probabilities = torch.softmax(self.model(state_tensor), dim=1)
        actions = torch.softmax(action_probabilities, dim=1)
        return actions.numpy()
    
import gymnasium as gym
import pandas as pd


class Environment(gym.Env):
    def __init__(
            self,
            data: pd.DataFrame,
            truncation: dict,
            randomize: bool = True,
            real_price_change_column: str = "price_change"
        ):
        self._data = data
        self.randomize = randomize
        self.real_price_change_column = real_price_change_column
        self.state_columns = [col for col in self._data.columns if col != self.real_price_change_column]
        self.truncation = truncation
        self.step_idx: int = 0
        
        self.action_space = gym.spaces.Discrete(2)    # -1 ~ Sell, 0 ~ Hold, 1 ~ Buy
        self.observation_space = gym.spaces.Box(0, 1, (len(self.state_columns),))
    
    def reward(self, action: int) -> float:
        bias = 1 if self.price_change > 0.0 else 0
        reward = 1 if bias == action else -1
        return reward, self.step_idx == self.truncation, 0

    @property
    def state(self):
        state = self.data[self.state_columns].iloc[self.step_idx].values
        return state
    
    @property
    def price_change(self):
        return self.data[self.real_price_change_column].iloc[self.step_idx]
    
    def reset(self):
        self.data = self._data.copy()
        if self.randomize:
            self.data = self.data.sample(frac=1).reset_index(drop=True)            
        self.step_idx = 0
        self.info = {"data_type": type(self.state)}
        return self.state, self.info
    
    def step(self, action: gym.spaces.Discrete):
        self.step_idx += 1      
        action -= 1
        reward, termination, truncation = self.reward(action=action)
        return self.state, reward, termination, truncation, self.info
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError
